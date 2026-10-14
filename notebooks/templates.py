"""Reusable temporal cross-validation and model-independent Optuna templates."""

from collections.abc import Callable, Iterator, Sequence
import hashlib
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor, Pool


class TimeSeriesCV:
    """Split unique calendar days; test_size, gap and max_train_size count observed days.

    ``split(frame)`` yields positional indices, including for unsorted input.
    ``prepare`` fits preprocessing separately on each training fold. Callbacks:
    fit_features(frame) -> (features, state), transform_features(frame, state) -> features.
    """

    def __init__(self, n_splits: int = 3, *, date_column: str = "date",
                 test_size: int | None = None, gap: int = 0,
                 max_train_size: int | None = None):
        self.date_column = date_column
        self.splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size,
                                       gap=gap, max_train_size=max_train_size)

    def split(self, frame: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(frame[self.date_column]).dt.normalize()
        if dates.isna().any():
            raise ValueError("Dates must not contain missing values")
        days = pd.DatetimeIndex(dates.unique()).sort_values()
        for train_days, valid_days in self.splitter.split(days):
            yield (np.flatnonzero(dates.isin(days[train_days])),
                   np.flatnonzero(dates.isin(days[valid_days])))

    def prepare(self, frame: pd.DataFrame, *, target: str,
                feature_columns: Sequence[str], fit_features: Callable,
                transform_features: Callable, metadata_columns: Sequence[str] = ()) -> list[dict]:
        """Build model-ready folds and expose date/row counts as ``summary_``."""
        folds, rows = [], []
        for number, (train_idx, valid_idx) in enumerate(self.split(frame), 1):
            train, valid = frame.iloc[train_idx].copy(), frame.iloc[valid_idx].copy()
            fit_frame, state = fit_features(train)
            valid_frame = transform_features(valid, state)
            for raw, features in [(train, fit_frame), (valid, valid_frame)]:
                if not raw.index.equals(features.index):
                    raise ValueError("Preprocessing must preserve row order and index")
            folds.append(dict(fold=number, X_train=fit_frame[list(feature_columns)],
                              X_valid=valid_frame[list(feature_columns)],
                              y_train=train[target], y_valid=valid[target],
                              valid_metadata=valid[list(metadata_columns)].copy(), feature_state=state))
            rows.append(dict(fold=number, train_start=train[self.date_column].min(),
                             train_end=train[self.date_column].max(),
                             valid_start=valid[self.date_column].min(),
                             valid_end=valid[self.date_column].max(),
                             train_rows=len(train), valid_rows=len(valid)))
        self.summary_ = pd.DataFrame(rows)
        return folds


class HParamsOpt:
    """Supply suggest(trial) -> params and evaluate(params, fold) -> metric dict.

    The metric dict must contain score_key; additional metrics are stored per fold.
    No predictions are pooled. ``run`` budgets completed trials across restarts.
    Use fingerprint() in study names to isolate changed data or configurations.
    """

    def __init__(self, folds: Sequence, suggest: Callable, evaluate: Callable, *,
                 weights: Sequence[float] | None = None, score_key: str = "score",
                 direction: str = "minimize", study_name: str | None = None,
                 storage: str | None = None, seed: int = 42):
        self.folds = list(folds)
        if not self.folds:
            raise ValueError("At least one fold is required")
        weights = np.ones(len(self.folds)) if weights is None else np.asarray(weights, dtype=float)
        if (weights.shape != (len(self.folds),) or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("Weights must be finite, nonnegative and match the folds")
        self.weights = weights / weights.sum()
        self.suggest, self.evaluate, self.score_key = suggest, evaluate, score_key
        self.study = optuna.create_study(
            study_name=study_name, direction=direction, storage=storage,
            sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=True)

    @staticmethod
    def fingerprint(config: dict, folds: Sequence[dict]) -> str:
        """Hash configuration plus prepared X/y values and indices for each fold."""
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for fold in folds:
            for part in ["train", "valid"]:
                for prefix in ["X", "y"]:
                    digest.update(pd.util.hash_pandas_object(
                        fold[f"{prefix}_{part}"], index=True).values.tobytes())
        return digest.hexdigest()[:16]

    def objective(self, trial: optuna.Trial) -> float:
        params = self.suggest(trial)
        results = [self.evaluate(params, fold) for fold in self.folds]
        scores = np.asarray([result[self.score_key] for result in results], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Fold scores must be finite")
        trial.set_user_attr(f"fold_{self.score_key}", scores.tolist())
        for key in results[0]:
            if key != self.score_key:
                trial.set_user_attr(key, [result[key] for result in results])
        return float(self.weights @ scores)

    def run(self, n_trials: int, *, initial_trials: Sequence[dict] | None = None, **kwargs) -> optuna.Study:
        """Run up to a total completed-trial budget; enqueue seed trials only once."""
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        if initial_trials is None:
            initial_trials = getattr(self, "initial_trials", ())
        if not self.study.trials:
            for params in initial_trials:
                self.study.enqueue_trial(params)
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in self.study.trials)
        options = dict(n_jobs=1, show_progress_bar=False, gc_after_trial=True)
        options.update(kwargs)
        self.study.optimize(self.objective, n_trials=max(0, n_trials - completed), **options)
        self.best_params_ = {**getattr(self, "fixed_params", {}), **self.study.best_params}
        self.selected_cv_score_ = float(self.study.best_value)
        return self.study


    @classmethod
    def catboost(cls, folds: Sequence[dict], *, cat_features: Sequence[str] = (),
                 weights: Sequence[float] | None = None, fixed_params: dict | None = None,
                 search_space: dict | None = None, storage: str | None = None,
                 study_prefix: str = "catboost", seed: int = 42,
                 log_target: bool = True, early_stopping_rounds: int = 100):
        """Configure the CatBoost adapter with conditional search and reusable fold pools.

        Defaults search growth policy, leaf estimation, column sampling and borders.
        Target transforms are inverted before scoring; no OOF predictions are retained.
        """
        fixed = dict(iterations=1500, loss_function="RMSE", eval_metric="RMSE",
                     random_seed=seed, thread_count=4, allow_writing_files=False)
        fixed.update(fixed_params or {})
        space = dict(depth=[4, 8], learning_rate=[.02, .15], l2_leaf_reg=[.1, 30.],
                     random_strength=[.001, 3.], colsample_bylevel=[.4, 1.],
                     border_count=[64, 128, 254],
                     grow_policy=["SymmetricTree", "Depthwise", "Lossguide"],
                     min_data_in_leaf=[1, 128], max_leaves=[16, 64],
                     leaf_estimation_method=["Newton", "Gradient"],
                     leaf_estimation_iterations=[1, 10],
                     leaf_estimation_backtracking=["No", "AnyImprovement"])
        space.update(search_space or {})
        fold_weights = np.ones(len(folds)) / len(folds) if weights is None else np.asarray(weights, dtype=float)
        config = dict(version=2, fixed=fixed, search=space, weights=fold_weights.tolist(),
                      feature_columns=folds[0]["X_train"].columns.tolist())
        if not log_target or early_stopping_rounds != 100:
            config.update(log_target=log_target, early_stopping_rounds=early_stopping_rounds)
        study_id = cls.fingerprint(config, folds)

        def suggest(trial):
            params = dict(fixed)
            for key in ["depth", "learning_rate", "l2_leaf_reg", "random_strength",
                        "colsample_bylevel", "border_count", "grow_policy",
                        "leaf_estimation_method", "leaf_estimation_iterations",
                        "leaf_estimation_backtracking", "min_data_in_leaf", "max_leaves"]:
                if key == "min_data_in_leaf" and params["grow_policy"] == "SymmetricTree":
                    continue
                if key == "max_leaves" and params["grow_policy"] != "Lossguide":
                    continue
                choices = space[key]
                if key in ["border_count", "grow_policy", "leaf_estimation_method", "leaf_estimation_backtracking"]:
                    params[key] = trial.suggest_categorical(key, choices)
                elif key in ["depth", "leaf_estimation_iterations", "min_data_in_leaf", "max_leaves"]:
                    params[key] = trial.suggest_int(key, *choices, log=key == "min_data_in_leaf")
                else:
                    params[key] = trial.suggest_float(key, *choices,
                        log=key in ["learning_rate", "l2_leaf_reg", "random_strength"])
            return params

        def evaluate(params, fold):
            model = optimizer._fit_fold(params, fold)
            predicted = optimizer.predict(model, fold["X_valid"])
            return {"mape_pct": float(100 * mean_absolute_percentage_error(fold["y_valid"], predicted)),
                    "tree_counts": model.tree_count_}

        optimizer = cls(folds, suggest, evaluate, weights=fold_weights, score_key="mape_pct",
                        study_name=f"{study_prefix}-{study_id}", storage=storage, seed=seed)
        optimizer.fixed_params = fixed
        optimizer.cat_features = list(cat_features)
        optimizer.log_target = log_target
        optimizer.early_stopping_rounds = early_stopping_rounds
        optimizer.feature_columns = folds[0]["X_train"].columns.tolist()
        optimizer._pools = {}
        for fold in folds:
            for part in ["train", "valid"]:
                y = optimizer._target(fold[f"y_{part}"])
                optimizer._pools[id(fold), part] = Pool(fold[f"X_{part}"], y, cat_features=cat_features)
        # Initial coverage of every growth / leaf-estimation pair.
        optimizer.initial_trials = [
            dict(depth=6, learning_rate=.05, l2_leaf_reg=3., random_strength=1.,
                 colsample_bylevel=1., border_count=254, grow_policy=policy,
                 leaf_estimation_method=method, leaf_estimation_iterations=1,
                 leaf_estimation_backtracking="No",
                 **({"min_data_in_leaf": 20} if policy != "SymmetricTree" else {}),
                 **({"max_leaves": 31} if policy == "Lossguide" else {}))
            for policy in space["grow_policy"] for method in space["leaf_estimation_method"]
        ] if search_space is None else []
        return optimizer

    def _target(self, values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (self.log_target and (values <= 0).any()):
            raise ValueError("Targets must be finite and positive for log transformation")
        return np.log(values) if self.log_target else values

    def _fit_fold(self, params: dict, fold: dict):
        model = CatBoostRegressor(**params)
        model.fit(self._pools[id(fold), "train"], eval_set=self._pools[id(fold), "valid"],
                  early_stopping_rounds=self.early_stopping_rounds, use_best_model=True, verbose=False)
        return model

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        """Predict in original target units with finite-value validation."""
        prediction = np.asarray(model.predict(X))
        if self.log_target:
            prediction = np.exp(prediction)
        if not np.isfinite(prediction).all() or (self.log_target and (prediction <= 0).any()):
            raise ValueError("Invalid model predictions")
        return prediction

    def fit(self, X: pd.DataFrame, y, *, params: dict | None = None, iterations: int | None = None):
        """Refit selected parameters on all supplied rows, without early stopping."""
        params = dict(self.best_params_ if params is None else params)
        if iterations is not None:
            params["iterations"] = iterations
        model = CatBoostRegressor(**params)
        model.fit(X, self._target(y), cat_features=self.cat_features, verbose=False)
        return model

    def fit_folds(self, params: dict | None = None) -> tuple[list, pd.DataFrame]:
        """Return fitted fold models and original-scale metrics, without OOF storage."""
        models, rows = [], []
        params = self.best_params_ if params is None else params
        for fold in self.folds:
            model = self._fit_fold(params, fold)
            prediction = self.predict(model, fold["X_valid"])
            rows.append(dict(fold=fold["fold"], iterations=model.tree_count_,
                MAE=mean_absolute_error(fold["y_valid"], prediction),
                RMSE=np.sqrt(mean_squared_error(fold["y_valid"], prediction)),
                MAPE_pct=100 * mean_absolute_percentage_error(fold["y_valid"], prediction)))
            models.append(model)
        return models, pd.DataFrame(rows)

    def tune_golden_features(self, golden_features: Sequence[str], *,
                             border_counts: Sequence[int] = (254, 512, 1024, 2048)) -> pd.DataFrame:
        """Sequentially select an independent border count for each named numeric feature.

        Selection uses weighted CV only. Feature indices always refer to the full
        input schema, including categorical columns. Empty list skips this stage.
        The current setting is included as a baseline so the score cannot regress.
        """
        features = list(golden_features)
        if len(features) != len(set(features)):
            raise ValueError("golden_features must be unique")
        for feature in features:
            if feature not in self.feature_columns or feature in self.cat_features:
                raise ValueError(f"Golden feature must be a known numeric column: {feature}")
            if not pd.api.types.is_numeric_dtype(self.folds[0]["X_train"][feature]):
                raise ValueError(f"Golden feature must be numeric: {feature}")
        counts = list(border_counts)
        if not counts or any(not isinstance(count, (int, np.integer)) or count < 1 for count in counts):
            raise ValueError("border_counts must contain positive integers")
        params = dict(self.best_params_)
        selected, rows, summary = {}, [], []
        for feature in features:
            trials = []
            candidates = sorted(set(counts + [int(params.get("border_count", 254))]))
            for count in candidates:
                settings = {**selected, feature: count}
                trial_params = {**params, "per_float_feature_quantization": [
                    f"{self.feature_columns.index(name)}:border_count={value}"
                    for name, value in settings.items()]}
                scores = []
                for fold in self.folds:
                    result = self.evaluate(trial_params, fold)
                    scores.append(result[self.score_key])
                    rows.append(dict(feature=feature, border_count=count, fold=fold["fold"],
                                     MAPE_pct=result[self.score_key], iterations=result["tree_counts"]))
                score = float(self.weights @ np.asarray(scores))
                trials.append((score, count))
                summary.append(dict(feature=feature, border_count=count, weighted_MAPE_pct=score))
            score, count = min(trials)
            selected[feature] = count
            self.selected_cv_score_ = score
        if selected:
            params["per_float_feature_quantization"] = [
                f"{self.feature_columns.index(name)}:border_count={count}" for name, count in selected.items()]
        self.best_params_ = params
        self.golden_borders_ = selected
        self.golden_metrics_ = pd.DataFrame(rows)
        self.golden_summary_ = pd.DataFrame(summary)
        return self.golden_summary_

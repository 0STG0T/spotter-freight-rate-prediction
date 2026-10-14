"""Checks for reusable temporal CV and optimization templates."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from templates import TimeSeriesCV, HParamsOpt


class TemplateTests(unittest.TestCase):
    def test_whole_days_and_gap_on_unsorted_rows(self):
        frame = pd.DataFrame({'date': pd.to_datetime([
            '2025-01-04', '2025-01-01', '2025-01-03',
            '2025-01-02', '2025-01-03 12:00', '2025-01-05'], format='mixed')})
        for train, valid in TimeSeriesCV(2, gap=1).split(frame):
            self.assertGreater((frame.iloc[valid].date.min().normalize()
                                - frame.iloc[train].date.max().normalize()).days, 1)
            self.assertFalse(set(frame.iloc[train].date.dt.normalize()) &
                             set(frame.iloc[valid].date.dt.normalize()))

    def test_preprocessing_is_fitted_only_on_each_train(self):
        frame = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=12),
                              'x': np.arange(12), 'y': np.arange(12)})
        def fit(rows):
            return rows, rows.x.mean()
        def transform(rows, mean):
            return rows.assign(x=rows.x - mean)
        cv = TimeSeriesCV(3)
        folds = cv.prepare(frame, target='y', feature_columns=['x'],
                           fit_features=fit, transform_features=transform)
        for fold, (train, valid) in zip(folds, cv.split(frame)):
            expected = frame.iloc[valid].x - frame.iloc[train].x.mean()
            pd.testing.assert_series_equal(fold['X_valid'].x, expected)

    def test_weighted_objective_and_resume_budget(self):
        optimizer = HParamsOpt([1., 3., 5.], lambda trial: {},
                              lambda params, fold: {'score': fold}, weights=[.2, .3, .5])
        study = optimizer.run(2)
        self.assertAlmostEqual(study.best_value, 3.6)
        optimizer.run(2)
        self.assertEqual(len(study.trials), 2)
        self.assertEqual(study.best_trial.user_attrs['fold_score'], [1., 3., 5.])

    def test_invalid_weights_and_dates(self):
        with self.assertRaises(ValueError):
            HParamsOpt([1, 2], lambda trial: {}, lambda params, fold: {}, weights=[1])
        with self.assertRaises(ValueError):
            list(TimeSeriesCV().split(pd.DataFrame({'date': [None]})))


    def test_catboost_golden_features_and_log_target(self):
        frame = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=40),
                              'city': ['a', 'b'] * 20, 'distance': np.arange(40) + 1.,
                              'weight': np.tile(np.arange(10), 4) + 1.})
        frame['y'] = 10. + 2 * frame.distance
        folds = TimeSeriesCV(2).prepare(
            frame, target='y', feature_columns=['city', 'distance', 'weight'],
            fit_features=lambda rows: (rows, None), transform_features=lambda rows, state: rows)
        optimizer = HParamsOpt.catboost(
            folds, cat_features=['city'], weights=[.3, .7],
            fixed_params={'iterations': 15, 'thread_count': 1},
            search_space={'depth': [2, 2], 'border_count': [8],
                          'grow_policy': ['SymmetricTree']})
        optimizer.run(1)
        with self.assertRaises(ValueError):
            optimizer.tune_golden_features(['city'])
        optimizer.tune_golden_features(['distance', 'weight'], border_counts=[4, 8])
        params = optimizer.best_params_['per_float_feature_quantization']
        self.assertEqual(params, [
            f'1:border_count={optimizer.golden_borders_["distance"]}',
            f'2:border_count={optimizer.golden_borders_["weight"]}'])
        models, metrics = optimizer.fit_folds()
        self.assertAlmostEqual(float(np.array([.3, .7]) @ metrics.MAPE_pct),
                               optimizer.selected_cv_score_, places=8)
        model = optimizer.fit(frame[['city', 'distance', 'weight']], frame.y, iterations=10)
        self.assertTrue((optimizer.predict(model, frame[['city', 'distance', 'weight']]) > 0).all())
        self.assertEqual(len(models), 2)


if __name__ == '__main__':
    unittest.main()

"""Shared data preparation, feature engineering, routing and plotting helpers."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import json
import os
import pickle
from importlib.metadata import version
import time

from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pandas.tseries.holiday import AbstractHolidayCalendar, USFederalHolidayCalendar
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, transform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# Data preparation, calendar and routing

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_FILES = {
    "train": "train-test.csv",
    "validation": "validation.csv",
    "december": "december-chart-inputs.csv",
}
INPUT_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal",
]
EXPECTED_COLUMNS = {
    "train": INPUT_COLUMNS + ["posted_rate"],
    "validation": INPUT_COLUMNS,
    "december": ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"],
}
METADATA_COLUMNS = ["date", "load_id", "posted_rate", "predicted_rate"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and mark invalid weights without dropping or sorting rows."""
    result = df.copy()
    for column in ["load_id", "pickup", "delivery", "equipment"]:
        if column in result:
            result[column] = result[column].astype("string").str.strip().replace("", pd.NA)
            if result[column].isna().any():
                raise ValueError(f"Missing {column}")
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
    if result["date"].isna().any():
        raise ValueError("Missing dates")
    numeric_columns = [
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
        "weight", "market_index", "quote_signal", "posted_rate", "predicted_rate",
    ]
    for column in numeric_columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="raise")
            if not np.isfinite(result[column].dropna()).all():
                raise ValueError(f"Non-finite {column}")
    result.loc[result["weight"].le(0), "weight"] = np.nan
    for column in ["distance", "quote_signal", "posted_rate"]:
        if column in result and (result[column].isna().any() or result[column].le(0).any()):
            raise ValueError(f"Missing or non-positive {column}")
    for column in ["pickup_lat", "delivery_lat", "pickup_lon", "delivery_lon"]:
        bound = 90 if column.endswith("_lat") else 180
        if column in result and (result[column].isna().any() or not result[column].between(-bound, bound).all()):
            raise ValueError(f"Invalid coordinate range: {column}")
    if "load_id" in result and result["load_id"].duplicated().any():
        raise ValueError("Duplicate load IDs")
    return result


def load_data(data_dir: str | Path = DATA_DIR) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read assessment inputs, validate their contracts, and normalize types."""
    data_dir = Path(data_dir)
    frames = {}
    for name, filename in SOURCE_FILES.items():
        df = pd.read_csv(data_dir / filename)
        if list(df.columns) != EXPECTED_COLUMNS[name]:
            raise ValueError(f"{name}: unexpected schema")
        frames[name] = clean_data(df)
    template = pd.read_csv(data_dir / "validation-predictions-template.csv")
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Unexpected prediction-template schema")
    if template["load_id"].isna().any() or template["load_id"].duplicated().any():
        raise ValueError("Invalid prediction-template IDs")
    train, validation, december = (frames[name] for name in SOURCE_FILES)
    expected_ids = {f"TE-{i:06d}" for i in range(1, 12_001)}
    if len(validation) != 12_000 or set(validation["load_id"]) != expected_ids:
        raise ValueError("Validation IDs do not match the scorer contract")
    if set(template["load_id"]) != expected_ids:
        raise ValueError("Prediction-template IDs differ from validation")
    if set(train["load_id"]) & set(validation["load_id"]):
        raise ValueError("Train and validation IDs overlap")
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Train and validation periods overlap")
    if len(december) != 31 or set(december["date"]) != set(pd.date_range("2025-12-01", "2025-12-31")):
        raise ValueError("Unexpected December dates")
    for column, expected in {"pickup": "Lexington", "delivery": "Fort Wayne", "equipment": "Dry Van",
                             "distance": 360, "weight": 32_000}.items():
        if not december[column].eq(expected).all():
            raise ValueError(f"Unexpected December {column}")
    return frames, template


def fit_imputation_values(train: pd.DataFrame) -> pd.Series:
    """Fit medians on the fitting partition only, including within local folds."""
    values = clean_data(train)[["weight", "market_index"]].median()
    if values.isna().any():
        raise ValueError("Cannot calculate training medians")
    return values


def prepare_data(df: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    """Fill existing predictor columns; leave prediction fields and absent columns alone."""
    result = clean_data(df)
    columns = [column for column in ["weight", "market_index"] if column in result]
    result[columns] = result[columns].fillna(fill_values)
    if result.drop(columns=["posted_rate", "predicted_rate"], errors="ignore").isna().any().any():
        raise ValueError("Unhandled missing predictor values")
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode weekly, monthly, and annual cycles without elapsed-time features."""
    result = df.copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing dates")
    phases = {
        "dow": dates.dt.dayofweek / 7,
        "dom": (dates.dt.day - 1) / dates.dt.days_in_month,
        "year": (dates.dt.dayofyear - 1) / np.where(dates.dt.is_leap_year, 366, 365),
    }
    for name, phase in phases.items():
        result[f"{name}_sin"] = np.sin(2 * np.pi * phase).astype("float32")
        result[f"{name}_cos"] = np.cos(2 * np.pi * phase).astype("float32")
    result["is_weekend"] = dates.dt.dayofweek.ge(5).astype("int8")
    return result


ROUTE_KEYS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
ROUTE_COLUMNS = ["road_distance_miles", "driving_time_hours", "road_to_air_ratio",
                 "input_to_road_ratio", "route_available"]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance for arrays or scalars, in miles."""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    value = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def attach_coordinates(df: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Recover absent December coordinates from supplied city mappings only."""
    result = df.copy()
    lookup = locations.set_index("city")
    for role in ["pickup", "delivery"]:
        for coord in ["lat", "lon"]:
            column = f"{role}_{coord}"
            if column not in result:
                result[column] = result[role].map(lookup[coord])
    if result[ROUTE_KEYS].isna().any().any():
        raise ValueError("Missing supplied coordinates for a route endpoint")
    return result


def add_route_features(df: pd.DataFrame, routes: pd.DataFrame,
                       allow_missing: bool = False) -> pd.DataFrame:
    """Join directed routes in original order; flag unresolved routes explicitly."""
    features = routes.set_index(ROUTE_KEYS)[["road_distance_miles", "driving_time_hours"]]
    if not features.index.is_unique:
        raise ValueError("Duplicate route keys")
    result = df.join(features, on=ROUTE_KEYS, validate="many_to_one")
    columns = ["road_distance_miles", "driving_time_hours"]
    values = result[columns]
    valid = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=result.index) & values.gt(0).all(axis=1)
    if not allow_missing and not valid.all():
        raise ValueError("Missing, unreachable or zero-length routes; review endpoints")
    result.loc[~valid, columns] = np.nan
    air = haversine_miles(*(result[column].to_numpy() for column in ROUTE_KEYS))
    result["road_to_air_ratio"] = result["road_distance_miles"] / np.where(air > 0, air, np.nan)
    result["input_to_road_ratio"] = result["distance"] / result["road_distance_miles"]
    result["route_available"] = valid.astype("int8")
    return result


def here_route_table(locations: pd.DataFrame, cache_dir: str | Path,
                     snapshot: str = "2026-10-12", batch_size: int = 10) -> pd.DataFrame:
    """Calculate worldwide truckFast routes; cache responses without credentials.

    Coordinates must be reviewed by the caller. HERE distances are meters and
    times are seconds. Cached blocks can be reused without an API key.
    """
    if not 1 <= batch_size <= 10:
        raise ValueError("HERE source batch_size must be between 1 and 10")
    points = locations[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    if points.empty or not np.isfinite(points.to_numpy()).all():
        raise ValueError("Invalid HERE routing points")
    if not points.lat.between(-90, 90).all() or not points.lon.between(-180, 180).all():
        raise ValueError("Invalid HERE coordinate bounds")
    config = {"schema": 1, "provider": "HERE-v8", "profile": "truckFast",
              "snapshot": snapshot, "points": points.to_dict("records")}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(cache_dir) / f"here_{fingerprint}"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(points), batch_size):
        sources = points.iloc[start:start + batch_size]
        # Bound request size even when the reference grows beyond this assessment.
        for target_start in range(0, len(points), 100):
            targets = points.iloc[target_start:target_start + 100]
            path = directory / f"matrix_{start}_{target_start}_{batch_size}.json"
            if path.exists():
                response = json.loads(path.read_text())
            else:
                api_key = os.environ.get("HERE_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("Set HERE_API_KEY in the kernel environment to calculate uncached routes.")
                payload = {"origins": sources.rename(columns={"lon": "lng"}).to_dict("records"),
                           "destinations": targets.rename(columns={"lon": "lng"}).to_dict("records"),
                           "profile": "truckFast", "regionDefinition": {"type": "world"},
                           "matrixAttributes": ["travelTimes", "distances"]}
                request = Request("https://matrix.router.hereapi.com/v8/matrix?async=false&apiKey=" + api_key,
                                  data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
                print(f"HERE: sources {start + 1}-{start + len(sources)}/{len(points)}, "
                      f"targets {target_start + 1}-{target_start + len(targets)}", flush=True)
                for attempt in range(4):
                    try:
                        with urlopen(request, timeout=180) as result:
                            response = json.load(result)
                        break
                    except HTTPError as exc:
                        if exc.code not in [429, 500, 502, 503, 504] or attempt == 3:
                            # Neither the authenticated URL nor credential-bearing response is printed.
                            raise RuntimeError(f"HERE Matrix HTTP {exc.code}; check app access, quota and request limits.") from None
                        retry_after = exc.headers.get("Retry-After", "")
                        time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1))
                    except (URLError, OSError, TimeoutError):
                        if attempt == 3:
                            raise RuntimeError("HERE Matrix network request failed; rerun to resume from cache.") from None
                        time.sleep(2 ** (attempt + 1))
                if "matrix" not in response:
                    raise RuntimeError("HERE returned no synchronous matrix; check service access and batch size.")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(response))
                temporary.replace(path)
                time.sleep(1.1)
            matrix = response["matrix"]
            expected = len(sources) * len(targets)
            if (matrix["numOrigins"] != len(sources) or matrix["numDestinations"] != len(targets)
                    or len(matrix["distances"]) != expected or len(matrix["travelTimes"]) != expected):
                raise ValueError("Unexpected HERE matrix dimensions")
            errors = matrix.get("errorCodes", [0] * expected)
            if len(errors) != expected:
                raise ValueError("Unexpected HERE error-code dimensions")
            for i, source in enumerate(sources.to_dict("records")):
                for j, target in enumerate(targets.to_dict("records")):
                    k = i * len(targets) + j
                    distance, seconds = matrix["distances"][k], matrix["travelTimes"][k]
                    if errors[k] or distance is None or seconds is None:
                        distance = hours = np.nan
                    else:
                        distance, hours = float(distance) / 1609.344, float(seconds) / 3600
                        if not np.isfinite([distance, hours]).all() or distance < 0 or hours < 0:
                            raise ValueError("Invalid HERE matrix values")
                    rows.append(dict(zip(ROUTE_KEYS, [source["lat"], source["lon"], target["lat"], target["lon"]]),
                                     road_distance_miles=distance, driving_time_hours=hours))
    return pd.DataFrame(rows)


def add_city_route_features(df: pd.DataFrame, centers: pd.DataFrame,
                            routes: pd.DataFrame) -> pd.DataFrame:
    """Use separate city-center endpoints; preserve the supplied coordinate fields."""
    result = df.drop(columns=ROUTE_COLUMNS, errors="ignore").copy()
    routing = result.drop(columns=ROUTE_KEYS, errors="ignore")
    routing = attach_coordinates(routing, centers)
    routed = add_route_features(routing, routes)
    for column in ROUTE_COLUMNS:
        result[column] = routed[column].to_numpy()
    return result


def here_route_geometry(origin: tuple[float, float], destination: tuple[float, float],
                        cache_dir: str | Path, snapshot: str) -> dict:
    """Fetch a road polyline using the same fast, static truck options as the matrix."""
    import flexpolyline

    params = {"origin": f"{origin[0]},{origin[1]}",
              "destination": f"{destination[0]},{destination[1]}",
              "transportMode": "truck", "routingMode": "fast", "departureTime": "any",
              "return": "polyline,summary"}
    signature = hashlib.sha256(json.dumps({"params": params, "snapshot": snapshot},
                                         sort_keys=True).encode()).hexdigest()[:20]
    path = Path(cache_dir) / f"here_shape_{signature}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        response = json.loads(path.read_text())
    else:
        api_key = os.environ.get("HERE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set HERE_API_KEY to calculate uncached route geometry.")
        request = Request("https://router.hereapi.com/v8/routes?" + urlencode({**params, "apiKey": api_key}))
        for attempt in range(4):
            try:
                with urlopen(request, timeout=120) as result:
                    response = json.load(result)
                break
            except HTTPError as exc:
                if exc.code not in [429, 500, 502, 503, 504] or attempt == 3:
                    raise RuntimeError(f"HERE route geometry HTTP {exc.code}") from None
                retry_after = exc.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1))
            except (URLError, OSError, TimeoutError):
                if attempt == 3:
                    raise RuntimeError("HERE route geometry network request failed.") from None
                time.sleep(2 ** (attempt + 1))
        if not response.get("routes"):
            raise ValueError("HERE returned no road route")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(response))
        temporary.replace(path)
        time.sleep(1.1)
    sections = response["routes"][0]["sections"]
    coordinates = [coordinate[:2] for section in sections
                   for coordinate in flexpolyline.decode(section["polyline"])]
    if len(coordinates) < 2 or not np.isfinite(coordinates).all():
        raise ValueError("Invalid HERE road geometry")
    return {"coordinates": coordinates,
            "distance_miles": sum(section["summary"]["length"] for section in sections) / 1609.344,
            "driving_hours": sum(section["summary"]["duration"] for section in sections) / 3600}


# Market index decomposition

MARKET_COEFFICIENT_COLUMNS = [
    'market_coef_intercept', 'market_coef_linear', 'market_coef_quadratic',
    'market_coef_weekly_sin', 'market_coef_weekly_cos',
]


def weekly_basis(dates):
    dates = pd.to_datetime(dates)
    weekday = np.asarray(dates.dt.dayofweek if isinstance(dates, pd.Series) else dates.dayofweek)
    phase = 2 * np.pi * weekday / 7
    return np.column_stack([np.sin(phase), np.cos(phase)])


@dataclass
class MarketIndexFeatures:
    """Fit daily training means, then evaluate the same curve for every load.

    A supplied curve_model may use known market inputs from inference rows.
    Residual reference and group statistics are fitted on training rows only.
    """
    curve_model: dict
    residual_reference: np.ndarray
    residual_mean: float
    residual_variance: float
    equipment_stats: pd.DataFrame
    lane_stats: pd.DataFrame

    @classmethod
    def fit(cls, train: pd.DataFrame, curve_model: dict | None = None):
        observed = train[['date', 'market_index']].dropna().copy()
        observed['date'] = pd.to_datetime(observed['date'])
        if not np.isfinite(observed.market_index).all():
            raise ValueError('Non-finite observed market index')
        model = select_market_curves(observed) if curve_model is None else curve_model
        residuals = train.loc[observed.index, ['equipment', 'pickup', 'delivery']].copy()
        residuals['residual'] = observed.market_index.to_numpy() - predict_market_curves(model, observed.date)
        reference = np.sort(residuals.residual.to_numpy())
        mean = float(reference.mean())
        variance = max(float(reference.var()), 1e-12)

        def group_stats(keys, strength):
            grouped = residuals.groupby(keys, dropna=False).residual.agg(['count', 'mean', 'var'])
            second = residuals.assign(square=residuals.residual ** 2).groupby(keys, dropna=False).square.mean()
            weight = grouped['count'] / (grouped['count'] + strength)
            center = weight * grouped['mean'] + (1 - weight) * mean
            moment = weight * second + (1 - weight) * (variance + mean ** 2)
            return pd.DataFrame({'count': grouped['count'], 'center': center,
                                 'scale': np.sqrt(np.maximum(moment - center ** 2, 1e-12)),
                                 'group_weight': weight})

        return cls(model, reference, mean, variance,
                   group_stats(['equipment'], 200), group_stats(['pickup', 'delivery'], 50))

    def coefficient_summary(self) -> pd.DataFrame:
        """Expose exact coefficients in the model's original time basis.

        t = days since origin / 365.25; weekly phase = 2*pi*weekday/7,
        where Monday is zero. Future dates use the same final segment formula.
        """
        return pd.DataFrame([
            {'start': segment['start'], 'end': segment['end'],
             'time_origin': self.curve_model['origin'],
             **dict(zip(MARKET_COEFFICIENT_COLUMNS, segment['coefficients'])),
             'weekly_amplitude': np.linalg.norm(segment['coefficients'][-2:])}
            for segment in self.curve_model['segments']
        ])

    def transform(self, frame: pd.DataFrame, observed_market: pd.Series | None = None):
        result = frame.copy()
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
        elapsed = (dates - self.curve_model['origin']).days.to_numpy() / 365.25
        segments = self.curve_model['segments']
        position = np.searchsorted([s['start'].value for s in segments[1:]], dates.asi8, side='right')
        coefficients = np.array([s['coefficients'] for s in segments])[position]
        for index, column in enumerate(MARKET_COEFFICIENT_COLUMNS):
            result[column] = coefficients[:, index]
        trend = predict_market_curves(self.curve_model, dates, weekly=False)
        expected = predict_market_curves(self.curve_model, dates)
        result['market_curve_trend'] = trend
        result['market_weekly_component'] = expected - trend
        result['market_curve_expected'] = expected
        # Derivatives inside each segment, in index/day and index/day squared.
        result['market_trend_slope'] = (coefficients[:, 1] + 2 * coefficients[:, 2] * elapsed) / 365.25
        result['market_trend_curvature'] = 2 * coefficients[:, 2] / 365.25 ** 2
        if observed_market is None:
            observed_market = frame.get('market_index', pd.Series(np.nan, index=frame.index))
        values = pd.to_numeric(observed_market.reindex(frame.index), errors='raise').to_numpy(dtype=float)
        available = np.isfinite(values)
        result['market_curve_residual'] = np.where(available, values - expected, 0)
        result['market_observed'] = available.astype('int8')
        residual = result.market_curve_residual.to_numpy()
        result['market_relative_residual'] = np.where(available, residual / np.maximum(np.abs(expected), 1e-6), 0)
        # Midrank empirical CDF, centered at zero; no distributional assumption.
        left = np.searchsorted(self.residual_reference, residual, side='left')
        right = np.searchsorted(self.residual_reference, residual, side='right')
        result['market_residual_percentile'] = np.where(available, (left + right) / len(self.residual_reference) - 1, 0)
        for name, keys, stats in [('equipment', ['equipment'], self.equipment_stats),
                                  ('lane', ['pickup', 'delivery'], self.lane_stats)]:
            group_index = (pd.Index(frame[keys[0]]) if len(keys) == 1
                           else pd.MultiIndex.from_frame(frame[keys]))
            matched = stats.reindex(group_index)
            center = matched.center.fillna(self.residual_mean).to_numpy()
            scale = matched.scale.fillna(np.sqrt(self.residual_variance)).to_numpy()
            result[f'market_{name}_residual_z'] = np.where(available, (residual - center) / scale, 0)
        return result


def fit_market_curves(train: pd.DataFrame, n_segments: int = 3, min_days: int = 7):
    """Fit piecewise quadratic segments plus a 7-day harmonic.

    Breakpoints minimize daily SSE on this fitting partition only. No target
    rates enter the fit. Known validation market inputs may be included explicitly.
    The resulting parabolas should
    not be assumed to extrapolate safely.
    """
    daily = train.groupby('date').market_index.mean().dropna().sort_index()
    daily.index = pd.to_datetime(daily.index)
    n = len(daily)
    if n_segments < 1 or n < n_segments * min_days:
        raise ValueError('Too few days for requested curve segments')
    elapsed = (daily.index - daily.index[0]).days.to_numpy() / 365.25
    x = np.column_stack([np.ones(n), elapsed, elapsed ** 2, weekly_basis(daily.index)])
    y = daily.to_numpy()
    a = np.concatenate([np.zeros((1, 5, 5)), np.cumsum(x[:, :, None] * x[:, None, :], axis=0)])
    b = np.vstack([np.zeros(5), np.cumsum(x * y[:, None], axis=0)])
    yy = np.r_[0, np.cumsum(y * y)]
    costs = np.full((n + 1, n + 1), np.inf)
    for left in range(n):
        for right in range(left + min_days, n + 1):
            coef = np.linalg.lstsq(a[right] - a[left], b[right] - b[left], rcond=None)[0]
            costs[left, right] = max(0, yy[right] - yy[left] - coef @ (b[right] - b[left]))
    loss = np.full((n_segments + 1, n + 1), np.inf); loss[0, 0] = 0
    parents = np.zeros((n_segments + 1, n + 1), dtype=int)
    for k in range(1, n_segments + 1):
        for right in range(k * min_days, n + 1):
            candidates = loss[k - 1, :right] + costs[:right, right]
            left = int(np.argmin(candidates))
            loss[k, right], parents[k, right] = candidates[left], left
    segments = []; right = n
    for k in range(n_segments, 0, -1):
        left = parents[k, right]
        coef = np.linalg.lstsq(x[left:right], y[left:right], rcond=None)[0]
        segments.append({'start': daily.index[left], 'end': daily.index[right - 1], 'coefficients': coef})
        right = left
    rss = max(float(loss[n_segments, n]), 1e-12)
    parameter_count = 5 * n_segments + n_segments - 1
    return {'origin': daily.index[0], 'segments': segments[::-1],
            'fit_rmse': np.sqrt(rss / n), 'bic': n * np.log(rss / n) + parameter_count * np.log(n)}


def predict_market_curves(model: dict, dates, weekly: bool = True):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    elapsed = (dates - model['origin']).days.to_numpy() / 365.25
    x = np.column_stack([np.ones(len(dates)), elapsed, elapsed ** 2, weekly_basis(dates)])
    if not weekly:
        x[:, -2:] = 0
    regime = np.searchsorted([s['start'].value for s in model['segments'][1:]], dates.asi8, side='right')
    return np.array([x[i] @ model['segments'][r]['coefficients'] for i, r in enumerate(regime)])


def select_market_curves(train: pd.DataFrame):
    """Choose 1-4 chronological segments by BIC on the supplied market observations."""
    observed_days = train.groupby('date').market_index.mean().notna().sum()
    max_segments = min(4, int(observed_days) // 7)
    if max_segments < 1:
        raise ValueError('Need at least 7 observed days for curve fitting')
    candidates = [fit_market_curves(train, n_segments=k) for k in range(1, max_segments + 1)]
    best = min(candidates, key=lambda model: model['bic'])
    best['selection'] = [{'segments': len(model['segments']), 'fit_rmse': model['fit_rmse'], 'bic': model['bic']}
                         for model in candidates]
    return best


# Holiday features

HOLIDAY_COLUMNS = [
    'is_us_holiday', 'is_us_observed_holiday', 'days_to_us_holiday',
    'days_since_us_holiday', 'is_before_us_holiday_3d',
    'is_after_us_holiday_3d', 'is_near_us_holiday_7d',
]


def add_holiday_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add seven date-only features, with distances capped at 14 days.

    Recurring federal holidays only, excluding state/local and exceptional
    executive closures. Federal closures are a calendar proxy for freight
    demand; they do not imply that private trucking businesses close.
    Distance/window features refer to observed dates; the holiday itself has
    distance zero and is excluded from the before/after windows.
    """
    result = frame.copy()
    if frame.empty:
        for column in HOLIDAY_COLUMNS:
            result[column] = pd.Series(index=frame.index, dtype='int8')
        return result
    dates = pd.to_datetime(frame.date).dt.normalize()
    if dates.isna().any():
        raise ValueError('Missing holiday feature dates')
    start = pd.Timestamp(year=dates.min().year - 1, month=1, day=1)
    end = pd.Timestamp(year=dates.max().year + 1, month=12, day=31)
    observed = USFederalHolidayCalendar().holidays(start, end).sort_values().unique()
    rules = deepcopy(USFederalHolidayCalendar.rules)
    for rule in rules:
        rule.observance = None
    actual = AbstractHolidayCalendar(rules=rules).holidays(start, end)
    result['is_us_holiday'] = dates.isin(actual).astype('int8')
    result['is_us_observed_holiday'] = dates.isin(observed).astype('int8')
    values = dates.to_numpy(dtype='datetime64[D]')
    holidays = observed.to_numpy(dtype='datetime64[D]')
    previous = np.searchsorted(holidays, values, side='right') - 1
    following = np.searchsorted(holidays, values, side='left')
    since = (values - holidays[previous]).astype(int)
    until = (holidays[following] - values).astype(int)
    result['days_to_us_holiday'] = np.minimum(until, 14).astype('int8')
    result['days_since_us_holiday'] = np.minimum(since, 14).astype('int8')
    result['is_before_us_holiday_3d'] = ((until >= 1) & (until <= 3)).astype('int8')
    result['is_after_us_holiday_3d'] = ((since >= 1) & (since <= 3)).astype('int8')
    result['is_near_us_holiday_7d'] = (np.minimum(since, until) <= 7).astype('int8')
    return result


# Historical city aggregates

AGGREGATE_COLUMNS = [
    'agg_pickup_share', 'agg_delivery_share', 'agg_pickup_balance',
    'agg_delivery_balance', 'agg_pickup_diversity', 'agg_delivery_diversity',
]


class PastCityAggregates:
    """Encode cities from earlier days; freeze training history for validation.

    Balance uses 50 smoothing observations by default. Shares describe the
    sample, not population demand. Cold-start cities receive zero features.
    """
    def __init__(self, strength=50):
        if strength <= 0:
            raise ValueError('Positive smoothing strength required')
        self.strength = strength

    @staticmethod
    def _inputs(frame):
        data = frame[['date', 'pickup', 'delivery']].copy()
        data['date'] = pd.to_datetime(data.date).dt.normalize()
        if data.isna().any().any() or not frame.index.is_unique:
            raise ValueError('Complete city/date inputs and unique indexes required')
        return data

    def _encode(self, data):
        values = []
        for row in data.itertuples():
            pickup_out, pickup_in = self.pickups.get(row.pickup, 0), self.deliveries.get(row.pickup, 0)
            delivery_out, delivery_in = self.pickups.get(row.delivery, 0), self.deliveries.get(row.delivery, 0)
            values.append([
                pickup_out / max(self.total, 1), delivery_in / max(self.total, 1),
                (pickup_out - pickup_in) / (pickup_out + pickup_in + self.strength),
                (delivery_out - delivery_in) / (delivery_out + delivery_in + self.strength),
                np.log1p(len(self.destinations.get(row.pickup, ()))),
                np.log1p(len(self.origins.get(row.delivery, ()))),
            ])
        return pd.DataFrame(values, index=data.index, columns=AGGREGATE_COLUMNS, dtype=float)

    def _update(self, data):
        for row in data.itertuples():
            self.pickups[row.pickup] += 1
            self.deliveries[row.delivery] += 1
            self.destinations[row.pickup].add(row.delivery)
            self.origins[row.delivery].add(row.pickup)
        self.total += len(data)

    def fit_transform(self, frame):
        self.total = 0
        self.pickups, self.deliveries = defaultdict(int), defaultdict(int)
        self.destinations, self.origins = defaultdict(set), defaultdict(set)
        data = self._inputs(frame)
        parts = []
        for _, day in data.groupby('date', sort=True):
            parts.append(self._encode(day))
            self._update(day)
        if not parts:
            raise ValueError('Non-empty fitting data required')
        return pd.concat(parts).reindex(frame.index)

    def transform(self, frame):
        if not hasattr(self, 'total') or not self.total:
            raise ValueError('Fit aggregates before transform')
        return self._encode(self._inputs(frame))


# Geographic diagnostics and maps

def audit_city_boundaries(frames: dict[str, pd.DataFrame], boundary_file: str | Path):
    """Check full-resolution polygons; boundary points count as inside (covers)."""
    collection = json.loads(Path(boundary_file).read_text())
    features = {feature['properties']['city']: feature for feature in collection['features']}
    if len(features) != len(collection['features']):
        raise ValueError('Duplicate city boundaries')
    parts = []
    for frame in frames.values():
        for role in ['pickup', 'delivery']:
            parts.append(frame[[role, f'{role}_lat', f'{role}_lon']].set_axis(['city', 'lat', 'lon'], axis=1))
    points = pd.concat(parts).drop_duplicates().sort_values('city')
    if points.city.duplicated().any():
        raise ValueError('City has multiple supplied coordinate pairs')
    rows = []
    for row in points.itertuples(index=False):
        feature = features[row.city]
        polygon = shape(feature['geometry'])
        if not polygon.is_valid or polygon.is_empty:
            raise ValueError(f'Invalid boundary: {row.city}')
        point = Point(row.lon, row.lat)
        inside = polygon.covers(point)
        # A local azimuthal equidistant projection centered at the tested point.
        # Radial distances from that point are measured in meters, not degrees.
        projection = f'+proj=aeqd +lat_0={row.lat} +lon_0={row.lon} +datum=WGS84 +units=m'
        forward = Transformer.from_crs('EPSG:4326', projection, always_xy=True)
        reverse = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True)
        projected = transform(forward.transform, polygon)
        projected_point = transform(forward.transform, point)
        nearest = nearest_points(projected_point, projected.boundary)[1]
        near_lon, near_lat = reverse.transform(nearest.x, nearest.y)
        rows.append({**feature['properties'], 'supplied_lat': row.lat, 'supplied_lon': row.lon,
                     'inside_named_city': inside,
                     'distance_to_boundary_km': projected_point.distance(projected.boundary) / 1000,
                     'distance_outside_city_km': 0.0 if inside else projected_point.distance(projected) / 1000,
                     'nearest_boundary_lat': near_lat, 'nearest_boundary_lon': near_lon})
    audit = pd.DataFrame(rows).sort_values('distance_outside_city_km', ascending=False).reset_index(drop=True)
    return audit, collection


def city_boundary_map(audit: pd.DataFrame, collection: dict,
                      state_file: str | Path):
    """Return a lightweight static map; no tile service or JavaScript renderer."""
    projector = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
    states = json.loads(Path(state_file).read_text())

    def outlines(features, tolerance):
        lines = []
        for feature in features:
            geometry = transform(projector.transform, shape(feature['geometry'])).simplify(tolerance)
            polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
            for polygon in polygons:
                lines.append(np.asarray(polygon.exterior.coords))
                lines.extend(np.asarray(ring.coords) for ring in polygon.interiors)
        return lines

    fig, ax = plt.subplots(figsize=(13, 7), dpi=130)
    ax.add_collection(LineCollection(outlines(states['features'], 1500),
                                     colors='#c5cbd1', linewidths=.55, zorder=1))
    ax.add_collection(LineCollection(outlines(collection['features'], 100),
                                     colors='#2463a6', linewidths=.8, zorder=2))
    for inside, color in [(False, '#d46b25'), (True, '#218c65')]:
        subset = audit.loc[audit.inside_named_city.eq(inside)]
        x, y = projector.transform(subset.supplied_lon.to_numpy(), subset.supplied_lat.to_numpy())
        ax.scatter(x, y, s=21 if not inside else 36, c=color,
                   edgecolors='white', linewidths=.5, zorder=3)
    # A few labels help orient the reader; all 72 cities remain in the table.
    for row in pd.concat([audit.nlargest(6, 'distance_outside_city_km'),
                          audit.loc[audit.inside_named_city]]).itertuples():
        x, y = projector.transform(row.supplied_lon, row.supplied_lat)
        offset = {'Phoenix': (5, 16), 'Los Angeles': (5, -13), 'Lubbock': (6, 12),
                  'San Antonio': (6, 0)}.get(row.city, (5, 5))
        ax.annotate(row.city, (x, y), xytext=offset, textcoords='offset points', fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 1}, zorder=4)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title('Original dataset points and official city boundaries', fontsize=15, pad=10)
    ax.legend(handles=[
        Line2D([], [], color='#2463a6', lw=1.2, label='Named-city boundaries'),
        Line2D([], [], marker='o', linestyle='none', color='#d46b25', label=f'Outside named city ({int((~audit.inside_named_city).sum())})'),
        Line2D([], [], marker='o', linestyle='none', color='#218c65', label=f'Inside named city ({int(audit.inside_named_city.sum())})'),
    ], loc='lower center', bbox_to_anchor=(.5, -.06), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .015, 'US Census TIGER/Line 2025. Original coordinates are unchanged. '
             'Outside the named city does not imply outside every city.', ha='center', fontsize=8, color='#666666')
    fig.subplots_adjust(left=.01, right=.99, top=.91, bottom=.1)
    return fig


def example_route_maps(routes: list[dict], state_file: str | Path):
    """Draw actual HERE road geometry as static panels, with local route extents."""
    projector = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
    states = json.loads(Path(state_file).read_text())
    outlines = []
    for feature in states['features']:
        geometry = transform(projector.transform, shape(feature['geometry'])).simplify(1500)
        polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
        outlines.extend(np.asarray(polygon.exterior.coords) for polygon in polygons)
    fig, axes = plt.subplots(1, len(routes), figsize=(15, 5), dpi=130, squeeze=False)
    colors = ['#2463a6', '#218c65', '#d46b25']
    for ax, route, color in zip(axes[0], routes, colors):
        ax.add_collection(LineCollection(outlines, colors='#c5cbd1', linewidths=.6, zorder=1))
        lat, lon = np.asarray(route['coordinates']).T
        x, y = projector.transform(lon, lat)
        ax.plot(x, y, color=color, lw=1.8, zorder=2)
        ax.scatter(x[0], y[0], marker='o', color=color, s=50, edgecolors='white', zorder=3)
        ax.scatter(x[-1], y[-1], marker='^', color=color, s=65, edgecolors='white', zorder=3)
        half_span = max(np.ptp(x), np.ptp(y), 100000) * .65
        mid_x, mid_y = (min(x) + max(x)) / 2, (min(y) + max(y)) / 2
        ax.set_xlim(mid_x - half_span, mid_x + half_span)
        ax.set_ylim(mid_y - half_span, mid_y + half_span)
        for city, px, py in [(route['pickup'], x[0], y[0]), (route['delivery'], x[-1], y[-1])]:
            ax.annotate(city, (px, py), xytext=(5, 6), textcoords='offset points', fontsize=8)
        ax.set_aspect('equal'); ax.set_axis_off()
        ax.set_title(f"{route['pickup']} to {route['delivery']}\n"
                     f"{route['distance_miles']:,.0f} miles | {route['driving_hours']:.1f} driving hours", fontsize=11)
    fig.suptitle('Example truck routes between city reference points', fontsize=15)
    fig.text(.5, .04, 'HERE road geometry. Circle: pickup; triangle: delivery. '
             'Static driving estimates; no loading or driver rest.', ha='center', fontsize=9, color='#666666')
    fig.tight_layout(rect=(0, .08, 1, .92))
    return fig


# Model explanations

def shap_summary(values: np.ndarray, features: pd.DataFrame, title: str,
                 max_display: int = 20, plot_rows: int = 1500, seed: int = 42):
    """Rank with all rows; display a deterministic sample of individual effects.

    Numeric colors use per-feature 5th/95th percentiles; categoricals are gray.
    SHAP values explain the model's raw log-rate prediction.
    """
    values = np.asarray(values)
    if values.shape != features.shape or not np.isfinite(values).all():
        raise ValueError('Finite SHAP matrix aligned with the feature frame required')
    order = np.argsort(np.abs(values).mean(axis=0))[-max_display:]
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(features), min(plot_rows, len(features)), replace=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('coolwarm')
    for y, column in enumerate(order):
        effects = values[sample, column]
        # Spread points vertically within local SHAP bins to show density.
        edges = np.linspace(effects.min(), effects.max() + 1e-12, 81)
        bins = np.clip(np.searchsorted(edges, effects, side='right') - 1, 0, 79)
        counts = np.bincount(bins, minlength=80)
        offsets = np.zeros(len(effects))
        for bucket in np.unique(bins):
            positions = np.flatnonzero(bins == bucket)
            width = .36 * np.sqrt(len(positions) / max(counts.max(), 1))
            offsets[positions] = rng.permutation(np.linspace(-width, width, len(positions))) if len(positions) > 1 else 0
        series = features.iloc[:, column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.to_numpy(dtype=float)
            low, high = np.nanquantile(numeric, [.05, .95])
            colors = np.full(len(sample), .5) if high <= low else np.clip((numeric[sample] - low) / (high - low), 0, 1)
            ax.scatter(effects, y + offsets, c=colors, cmap=cmap, vmin=0, vmax=1,
                       s=9, alpha=.6, linewidths=0, rasterized=True)
        else:
            ax.scatter(effects, y + offsets, color='#888888', s=9, alpha=.6,
                       linewidths=0, rasterized=True)
    ax.axvline(0, color='#888888', lw=.8)
    ax.set_yticks(range(len(order)), features.columns[order])
    ax.set(title=title, xlabel='SHAP value: contribution to ln(rate)')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, pad=.02, fraction=.025)
    colorbar.set_ticks([0, 1], labels=['Low', 'High'])
    colorbar.set_label('Numeric feature value')
    fig.text(.5, .01, f'Ranking: all {len(features):,} rows | Points: {len(sample):,} rows | Gray: categorical features',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .03, 1, 1])
    return fig


# Modeling diagnostics

def plot_market_decomposition(market_encoder, development_raw, holdout_raw, validation_raw,
                              holdout_start, output_path):
    """Plot observed daily market fit, trend, harmonic and segment equations."""
    market_curves = market_encoder.curve_model
    fig = plt.figure(figsize=(14, 11), layout='constrained')
    grid = fig.add_gridspec(4, 1, height_ratios=[2.1, 1.4, 1.3, 1.65])
    axes = [fig.add_subplot(grid[0])]
    axes.extend(fig.add_subplot(grid[i], sharex=axes[0]) for i in [1, 2])
    equations = fig.add_subplot(grid[3])
    equations.axis('off')
    segment_colors = ['#2463a6', '#2a8b57', '#b76b14', '#9557a5']
    validation_start = holdout_start
    plot_end = validation_raw.date.max()
    for name, frame, style in [('development', development_raw, '-'),
                               ('test', holdout_raw, ':'),
                               ('validation', validation_raw, ':')]:
        daily_mean = frame.groupby('date').market_index.mean().dropna()
        axes[0].plot(daily_mean.index, daily_mean, color='#666666', lw=1,
                     alpha=.65, linestyle=style, label=f'{name}: daily mean')

    for i, segment in enumerate(market_curves['segments']):
        color = segment_colors[i % len(segment_colors)]
        last = i == len(market_curves['segments']) - 1
        end = segment['end']
        dates = pd.date_range(segment['start'], end)
        trend = predict_market_curves(market_curves, dates, weekly=False)
        expected = predict_market_curves(market_curves, dates)
        for ax, values in zip(axes, [expected, trend, expected - trend]):
            ax.axvspan(segment['start'], end + pd.Timedelta(days=1), color=color, alpha=.055)
            ax.plot(dates, values, color=color, lw=1.5)
            if i:
                ax.axvline(segment['start'], color=color, lw=.8, alpha=.6)
        center = segment['start'] + (end - segment['start']) / 2
        axes[1].text(center, .96, f'Curve {i + 1}', color=color,
                     ha='center', va='top', transform=axes[1].get_xaxis_transform(), fontsize=10)
        a, b, c, u, v = segment['coefficients']
        formula = (f'$f_{i+1}(t) = {a:.5f} {b:+.5f}t {c:+.5f}t^2 '
                   f'{u:+.5f}\\sin(\\theta) {v:+.5f}\\cos(\\theta)$')
        period = f"{segment['start']:%b %d} – {segment['end']:%b %d}"
        row_y = .82 - i * .20
        equations.text(.01, row_y, f'Curve {i+1}: {period}', color=color, fontsize=9, va='center')
        equations.text(.36, row_y, formula, color=color, fontsize=10, va='center')

    for ax, title in zip(axes, ['Observed daily mean market index and piecewise fit', 'Quadratic trend', 'Weekly component']):
        ax.set(title=title, ylabel='Market index')
        ax.axvline(validation_start, color='#333333', linestyle=':', lw=1)
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].text(.99, .95, 'Fit uses observed train + validation market index', transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=9)
    axes[-1].set_xlabel('Date (2025)')
    origin = market_curves['origin']
    equations.set_title(
        f'Fixed coefficients per segment; t = days since {origin:%Y-%m-%d} / 365.25; '
        'theta = 2*pi*weekday/7 (Monday = 0)', fontsize=10, loc='left')
    market_plot = Path(output_path)
    market_plot.parent.mkdir(exist_ok=True)
    fig.savefig(market_plot, dpi=150, facecolor='white')
    plt.show()
    plt.close(fig)


def plot_time_series_folds(fold_summary, development_raw, holdout_raw, holdout_start, output_path):
    """Plot expanding-window folds and the separate held-out period."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    colors = {'train': '#2463a6', 'valid': '#d46b25'}
    for row in fold_summary.itertuples(index=False):
        for part in ['train', 'valid']:
            start = getattr(row, f'{part}_start')
            end = getattr(row, f'{part}_end') + pd.Timedelta(days=1)
            ax.broken_barh([(mdates.date2num(start), (end - start).days)],
                           (row.fold - .3, .6), facecolors=colors[part])
            count = getattr(row, f'{part}_rows')
            center = start + (end - start) / 2
            ax.text(center, row.fold, f'{count:,} loads', ha='center', va='center',
                    color='white', fontsize=9)
    for frame, color in [(development_raw, colors['train']), (holdout_raw, '#2a8b57')]:
        start, end = frame.date.min(), frame.date.max() + pd.Timedelta(days=1)
        ax.broken_barh([(mdates.date2num(start), (end - start).days)], (3.7, .6), facecolors=color)
        ax.text(start + (end - start) / 2, 4, f'{len(frame):,}', ha='center', va='center', color='white', fontsize=8)
    ax.axvline(holdout_start, color='#2a8b57', linestyle=':', linewidth=1)
    ax.set_yticks([1, 2, 3, 4], ['Fold 1', 'Fold 2', 'Fold 3', 'Test model'])
    ax.set_ylim(4.6, .4)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set(title='Three development folds and a 14-day held-out test', xlabel='2025')
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.2)
    ax.legend(handles=[Patch(color=colors['train'], label='Train'),
                       Patch(color=colors['valid'], label='CV validation'),
                       Patch(color='#2a8b57', label='Held-out test')],
              loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=3, frameon=False)
    fig.tight_layout()
    cv_plot = Path(output_path)
    cv_plot.parent.mkdir(exist_ok=True)
    fig.savefig(cv_plot, dpi=150, facecolor='white')
    plt.show()
    plt.close(fig)



def test_residuals(actual: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep load metadata and rank errors in original target units and percent."""
    if not actual.index.equals(predictions.index) or not actual.load_id.equals(predictions.load_id):
        raise ValueError("Test prediction rows must match actual rows exactly")
    result = actual.copy()
    result['predicted_rate'] = predictions.predicted_rate
    if not np.isfinite(result[['posted_rate', 'predicted_rate']].to_numpy()).all():
        raise ValueError("Non-finite rates in residual analysis")
    if result.posted_rate.le(0).any():
        raise ValueError("Relative errors require positive actual rates")
    result['residual'] = result.posted_rate - result.predicted_rate
    result['absolute_error'] = result.residual.abs()
    result['relative_error_pct'] = 100 * result.residual / result.posted_rate
    result['APE_pct'] = result.relative_error_pct.abs()
    return result


def plot_test_residuals(residuals: pd.DataFrame):
    """Inspect signed errors, magnitude effects, calibration and temporal bias."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    axes[0, 0].hist(residuals.relative_error_pct, bins=60, color='#2463a6', alpha=.8)
    axes[0, 0].axvline(0, color='black', lw=.8)
    axes[0, 0].set(title='Signed relative error distribution', xlabel='100 * (actual - predicted) / actual', ylabel='Loads')
    axes[0, 1].scatter(residuals.predicted_rate, residuals.residual, s=9, alpha=.35)
    axes[0, 1].axhline(0, color='black', lw=.8)
    axes[0, 1].set(title='Residual vs. predicted rate', xlabel='Predicted rate', ylabel='Actual - predicted')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('symlog', linthresh=100)
    axes[1, 0].scatter(residuals.posted_rate, residuals.predicted_rate, s=9, alpha=.35)
    bounds = [residuals[['posted_rate', 'predicted_rate']].min().min(),
              residuals[['posted_rate', 'predicted_rate']].max().max()]
    axes[1, 0].plot(bounds, bounds, color='black', lw=.8)
    axes[1, 0].set(title='Actual vs. predicted rate', xlabel='Actual rate', ylabel='Predicted rate', xscale='log', yscale='log')
    daily = residuals.groupby('date').agg(bias_pct=('relative_error_pct', 'mean'), MAPE_pct=('APE_pct', 'mean'))
    axes[1, 1].plot(daily.index, daily.bias_pct, marker='o', ms=3, label='Mean signed error')
    axes[1, 1].plot(daily.index, daily.MAPE_pct, marker='o', ms=3, label='MAPE')
    axes[1, 1].axhline(0, color='black', lw=.8)
    axes[1, 1].set(title='Daily error and bias', ylabel='Percent')
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[1, 1].tick_params(axis='x', rotation=30)
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('Held-out test residuals; positive residual = underprediction')
    return fig


def worst_prediction_shap(model, X: pd.DataFrame, worst: pd.DataFrame,
                          cat_features, max_display: int = 20):
    """Decision paths for worst test predictions, using the actual test model.

    Native SHAP contributions accumulate from expected log(rate) to each
    predicted log(rate). Remaining features are summed, preserving additivity.
    """
    from catboost import Pool

    selected = X.loc[worst.index]
    explanation = model.get_feature_importance(Pool(selected, cat_features=cat_features),
                                               type='ShapValues', thread_count=4)
    values, bases = explanation[:, :-1], explanation[:, -1]
    predictions = np.asarray(model.predict(selected, prediction_type='RawFormulaVal'))
    if not np.allclose(values.sum(axis=1) + bases, predictions, rtol=1e-6, atol=1e-6):
        raise ValueError('Native SHAP failed additivity for worst test predictions')
    order = np.argsort(np.abs(values).mean(axis=0))[::-1]
    shown = order[:max_display]
    hidden = order[max_display:]
    names = [str(selected.columns[i]) for i in shown[::-1]]
    contributions = values[:, shown[::-1]]
    if len(hidden):
        names.insert(0, f'Other {len(hidden)} features (sum)')
        contributions = np.column_stack([values[:, hidden].sum(axis=1), contributions])
    paths = np.column_stack([bases, bases[:, None] + np.cumsum(contributions, axis=1)])
    fig, ax = plt.subplots(figsize=(12, 9), layout='constrained')
    for i, (_, row) in enumerate(worst.iterrows()):
        label = (f'{row.load_id}: actual={row.posted_rate:.0f}, predicted={row.predicted_rate:.0f}, '
                 f'APE={row.APE_pct:.1f}%')
        line, = ax.plot(paths[i], np.arange(len(names) + 1), marker='o', ms=3, lw=1.5, label=label)
        color = line.get_color()
        comparison_y = len(names) + 1 + i
        actual_log = np.log(row.posted_rate)
        ax.vlines(predictions[i], len(names), comparison_y, color=color, lw=.8, alpha=.4, linestyle=':')
        ax.hlines(comparison_y, actual_log, predictions[i], color=color, lw=1.2, linestyle='--')
        ax.scatter([actual_log], [comparison_y], marker='x', s=65, linewidths=2, color=color, zorder=5)
        ax.scatter([predictions[i]], [comparison_y], marker='o', s=30, color=color, zorder=5)
    ax.set_yticks(np.arange(len(names) + 1 + len(worst)),
                 ['Expected log(rate)'] + names + [f'{load_id}: actual / prediction' for load_id in worst.load_id])
    ax.set(title='Five worst test predictions: SHAP decision paths\nX = actual rate; circle = predicted rate (both on log scale)',
           xlabel='Cumulative SHAP value + base value = predicted log(rate)')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.08), fontsize=8)
    details = pd.DataFrame(values, index=selected.index, columns=selected.columns)
    details['base_value'] = bases
    details['predicted_log_rate'] = predictions
    return fig, details


def save_model_bundle(model, directory: str | Path, *, feature_state: dict,
                      feature_columns, cat_features, city_centers: pd.DataFrame,
                      route_table: pd.DataFrame, metadata: dict,
                      verification_features: pd.DataFrame | None = None) -> Path:
    """Persist a native CatBoost model, fitted preprocessing and inference schema.

    Load preprocessing.pkl with notebooks/ on sys.path (it contains utils classes).
    Route references are bundled so inference can reuse them without network calls.
    """
    from catboost import CatBoostRegressor

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / 'model.cbm'
    temporary = directory / 'model.tmp.cbm'
    model.save_model(str(temporary))
    if verification_features is not None:
        restored = CatBoostRegressor()
        restored.load_model(str(temporary))
        if not np.allclose(model.predict(verification_features), restored.predict(verification_features),
                           rtol=1e-12, atol=1e-12):
            raise ValueError('Saved model predictions differ after reload')
    temporary.replace(model_path)
    state = dict(feature_state=feature_state, feature_columns=list(feature_columns),
                 cat_features=list(cat_features), city_centers=city_centers, route_table=route_table)
    temporary = directory / 'preprocessing.tmp.pkl'
    temporary.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(directory / 'preprocessing.pkl')
    manifest = dict(metadata, feature_columns=list(feature_columns), cat_features=list(cat_features),
                    model_file='model.cbm', preprocessing_file='preprocessing.pkl',
                    model_params=model.get_all_params(),
                    versions={name: version(name) for name in ['catboost', 'numpy', 'pandas', 'scikit-learn']})
    temporary = directory / 'metadata.tmp.json'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(directory / 'metadata.json')
    return directory


def compare_test_outliers(residuals: pd.DataFrame, development: pd.DataFrame,
                          features: pd.DataFrame, threshold: float = 50):
    """Describe high-APE test loads; target-derived ratios are diagnostic only."""
    result = residuals.copy()
    for column in ['road_distance_miles', 'driving_time_hours', 'road_to_air_ratio',
                   'input_to_road_ratio', 'market_curve_residual']:
        result[column] = features.loc[result.index, column]
    lane_counts = development.groupby(['pickup', 'delivery']).size()
    result['train_lane_count'] = lane_counts.reindex(
        pd.MultiIndex.from_frame(result[['pickup', 'delivery']])).fillna(0).to_numpy()
    result['quote_total'] = result.quote_signal * result.distance
    result['target_to_quote'] = result.posted_rate / result.quote_total
    result['prediction_to_quote'] = result.predicted_rate / result.quote_total
    result['actual_to_prediction'] = result.posted_rate / result.predicted_rate
    result['error_group'] = np.select(
        [result.APE_pct.le(10), result.APE_pct.le(threshold), result.residual.lt(0)],
        ['APE <= 10%', f'10% < APE <= {threshold:g}%', 'Severe: overprediction'],
        default='Severe: underprediction')
    summary = result.groupby('error_group', sort=False).agg(
        rows=('load_id', 'size'), MAPE_pct=('APE_pct', 'mean'),
        median_actual=('posted_rate', 'median'), median_prediction=('predicted_rate', 'median'),
        median_target_to_quote=('target_to_quote', 'median'),
        median_actual_to_prediction=('actual_to_prediction', 'median'))
    summary['row_share_pct'] = 100 * summary.rows / len(result)
    summary['share_of_total_APE_pct'] = 100 * result.groupby('error_group').APE_pct.sum() / result.APE_pct.sum()
    summary['unseen_lane_pct'] = 100 * result.train_lane_count.eq(0).groupby(result.error_group).mean()
    columns = ['distance', 'road_distance_miles', 'weight', 'market_index', 'quote_signal',
               'road_to_air_ratio', 'input_to_road_ratio', 'market_curve_residual',
               'train_lane_count', 'posted_rate', 'target_to_quote', 'prediction_to_quote']
    medians = result.groupby('error_group')[columns].median().T
    severe = result.APE_pct.gt(threshold)
    categorical = []
    for column in ['equipment', 'pickup', 'delivery']:
        a = result.loc[severe, column].value_counts()
        b = result.loc[~severe, column].value_counts()
        table = pd.DataFrame({'severe_rows': a, 'other_rows': b}).fillna(0)
        table['severe_share_pct'] = 100 * table.severe_rows / max(severe.sum(), 1)
        table['other_share_pct'] = 100 * table.other_rows / max((~severe).sum(), 1)
        table['difference_pp'] = table.severe_share_pct - table.other_share_pct
        table = table.sort_values('difference_pp', ascending=False).head(5)
        table.index.name = 'category'
        categorical.append(table.reset_index().assign(feature=column))
    # Similar historical loads: same equipment; closest standardized input vectors.
    # Matching never uses posted_rate or the prediction/error.
    match_columns = ['distance', 'weight', 'quote_signal', 'market_index']
    reference = development.copy()
    center = reference[match_columns].median()
    scale = (reference[match_columns].quantile(.75) - reference[match_columns].quantile(.25)).replace(0, 1)
    reference[match_columns] = reference[match_columns].fillna(center)
    peers = []
    for _, row in result.nlargest(5, 'APE_pct').iterrows():
        pool = reference.loc[reference.equipment.eq(row.equipment)]
        query = row[match_columns].astype(float).fillna(center)
        distance = (((pool[match_columns] - query) / scale) ** 2).sum(axis=1)
        closest = pool.loc[distance.nsmallest(20).index]
        peer_median = closest.posted_rate.median()
        peers.append(dict(load_id=row.load_id, pickup=row.pickup, delivery=row.delivery,
                          equipment=row.equipment, distance=row.distance, quote_signal=row.quote_signal,
                          actual=row.posted_rate, predicted=row.predicted_rate,
                          APE_pct=row.APE_pct, historical_peers=len(closest),
                          peer_rate_median=peer_median, peer_rate_p10=closest.posted_rate.quantile(.1),
                          peer_rate_p90=closest.posted_rate.quantile(.9),
                          actual_to_peer_median=row.posted_rate / peer_median))
    return result, summary, medians, pd.concat(categorical, ignore_index=True), pd.DataFrame(peers)


def plot_outlier_comparison(comparison: pd.DataFrame, threshold: float = 50):
    """Contrast ordinary and severe test errors against the observed quote input."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    colors = ['#9ba6b0', '#a8c8e6', '#d64c40', '#8755ad']
    groups = ['APE <= 10%', f'10% < APE <= {threshold:g}%',
              'Severe: overprediction', 'Severe: underprediction']
    for group, color in zip(groups, colors):
        part = comparison.loc[comparison.error_group.eq(group)]
        for ax, column in zip(axes, ['posted_rate', 'predicted_rate']):
            ax.scatter(part.quote_total, part[column], s=22 if group.startswith('Severe') else 8,
                       alpha=.85 if group.startswith('Severe') else .3, color=color, label=f'{group} (n={len(part)})')
    for ax, title in zip(axes, ['Actual rate vs. quote × distance', 'Model prediction vs. quote × distance']):
        ax.set(title=title, xlabel='Quote signal × distance', ylabel='Rate', xscale='log', yscale='log')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    return fig


def load_model_bundle(directory: str | Path):
    """Load a saved native model, preprocessing state and inference metadata."""
    from catboost import CatBoostRegressor

    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    state = pickle.loads((directory / metadata['preprocessing_file']).read_bytes())
    model = CatBoostRegressor().load_model(str(directory / metadata['model_file']))
    if list(model.feature_names_) != state['feature_columns']:
        raise ValueError('Saved model feature order differs from preprocessing schema')
    return model, state, metadata


def prepare_inference_features(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Transform inputs using saved statistics only; targets are never required."""
    frame = frame.drop(columns=['posted_rate', 'predicted_rate'], errors='ignore')
    state = bundle['feature_state']
    result = add_time_features(prepare_data(frame, state['fill_values']))
    result = add_city_route_features(result, bundle['city_centers'], bundle['route_table'])
    observed_market = frame.get('market_index', pd.Series(np.nan, index=frame.index))
    result = state['market_encoder'].transform(result, observed_market)
    result = add_holiday_features(result)
    result = result.join(state['city_encoder'].transform(result))
    result = result[bundle['feature_columns']]
    if not result.index.equals(frame.index) or result.isna().any().any():
        raise ValueError('Invalid inference feature alignment or missing values')
    return result


def predict_saved_model(model, frame: pd.DataFrame, bundle: dict, metadata: dict) -> pd.DataFrame:
    """Run saved preprocessing and invert the persisted target transformation."""
    features = prepare_inference_features(frame, bundle)
    prediction = np.asarray(model.predict(features))
    transform = metadata['prediction_transform']
    if transform == 'exp':
        prediction = np.exp(prediction)
    elif transform != 'identity':
        raise ValueError(f'Unsupported prediction transform: {transform}')
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('Rates must be finite and positive')
    return frame[['load_id', 'date']].assign(predicted_rate=prediction)


def plot_inference_predictions(validation: pd.DataFrame):
    """Describe predictions for supplied inputs with no observed rate targets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    rates = validation.predicted_rate
    axes[0, 0].hist(rates, bins=np.geomspace(rates.min(), rates.max(), 45), color='#2463a6', alpha=.8)
    axes[0, 0].set(title='Validation: predicted rate distribution', xlabel='Predicted rate', ylabel='Loads', xscale='log')
    daily = validation.groupby('date').predicted_rate.agg(
        mean='mean', p10=lambda x: x.quantile(.1), p90=lambda x: x.quantile(.9))
    axes[0, 1].plot(daily.index, daily['mean'], label='Mean predicted rate')
    axes[0, 1].fill_between(daily.index, daily.p10, daily.p90, alpha=.15, label='P10–P90 across loads')
    axes[0, 1].set(title='Validation: daily predictions', ylabel='Predicted rate')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[0, 1].tick_params(axis='x', rotation=30)
    for equipment, part in validation.groupby('equipment'):
        axes[1, 0].scatter(part.distance, part.predicted_rate, s=8, alpha=.3, label=equipment)
    axes[1, 0].set(title='Predicted rate vs. input distance', xlabel='Distance (miles)', ylabel='Predicted rate')
    axes[1, 0].legend(fontsize=8)
    groups = list(validation.groupby('equipment'))
    axes[1, 1].boxplot([part.predicted_rate for _, part in groups], showfliers=False)
    axes[1, 1].set_xticks(range(1, len(groups) + 1), [name for name, _ in groups])
    axes[1, 1].set(title='Predicted rate by equipment (fliers hidden)', ylabel='Predicted rate')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Supplied validation: predictions only; actual rates are unavailable')
    return fig


def prepare_december_observations(december: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Copy city coordinates and same-date/route/equipment market observations.

    No interpolation, averaging or fallback: missing and conflicting values stay
    missing. Source load IDs identify matching rows, even if a value is ambiguous.
    """
    result = december.copy()
    result['date'] = pd.to_datetime(result.date)
    reference = reference.copy()
    reference['date'] = pd.to_datetime(reference.date)
    for role in ['pickup', 'delivery']:
        for city in result[role].unique():
            points = pd.concat([
                reference.loc[reference[side].eq(city), [f'{side}_lat', f'{side}_lon']]
                    .set_axis(['lat', 'lon'], axis=1)
                for side in ['pickup', 'delivery']
            ]).drop_duplicates()
            if len(points) != 1 or points.isna().any().any():
                raise ValueError(f'Expected one observed coordinate pair for {city}')
            for coord in ['lat', 'lon']:
                result.loc[result[role].eq(city), f'{role}_{coord}'] = float(points.iloc[0][coord])
    for column in ['market_index', 'quote_signal']:
        result[column] = np.nan
    result['matched_load_ids'] = ''
    result['matched_load_count'] = 0
    for index, row in result.iterrows():
        matched = reference.loc[
            reference.date.eq(row.date) & reference.pickup.eq(row.pickup)
            & reference.delivery.eq(row.delivery) & reference.equipment.eq(row.equipment)]
        result.loc[index, 'matched_load_ids'] = ', '.join(matched.load_id.astype(str))
        result.loc[index, 'matched_load_count'] = len(matched)
        for column in ['market_index', 'quote_signal']:
            values = matched[column].dropna().unique()
            if len(values) == 1:
                result.loc[index, column] = values[0]
    return result


def fill_linear_quote_signal(frame: pd.DataFrame):
    """Fit a date-based least-squares line and fill only missing quote observations."""
    result = frame.copy()
    dates = pd.to_datetime(result.date)
    origin = dates.min()
    elapsed = (dates - origin).dt.total_seconds().to_numpy() / 86400
    observed = (result.quote_signal_observed.astype(bool).to_numpy()
                if 'quote_signal_observed' in result else result.quote_signal.notna().to_numpy())
    if np.unique(elapsed[observed]).size < 2:
        raise ValueError('At least two distinct observed quote dates are required')
    slope, intercept = np.polyfit(elapsed[observed], result.loc[observed, 'quote_signal'], 1)
    fitted = intercept + slope * elapsed
    if not np.isfinite(fitted).all() or (fitted <= 0).any():
        raise ValueError('Linear quote fit produces invalid rates')
    result['quote_signal_observed'] = observed
    result['quote_signal_linear_fit'] = fitted
    result.loc[~observed, 'quote_signal'] = fitted[~observed]
    outside = (elapsed < elapsed[observed].min()) | (elapsed > elapsed[observed].max())
    result['quote_signal_source'] = np.where(observed, 'observed validation',
                                            np.where(outside, 'linear extrapolation', 'linear interpolation'))
    fit = dict(origin=origin, intercept=float(intercept), slope_per_day=float(slope),
               observed_points=int(observed.sum()), filled_points=int((~observed).sum()))
    return result, fit


def plot_quote_signal_fit(frame: pd.DataFrame, fit: dict):
    """Show the fitted line, original observations and filled values separately."""
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    observed = frame.quote_signal_observed
    ax.plot(frame.date, frame.quote_signal_linear_fit, color='#2463a6', lw=1.5,
            label=f'Least-squares fit: {fit["intercept"]:.5f} + {fit["slope_per_day"]:.5f} × day')
    ax.scatter(frame.loc[~observed, 'date'], frame.loc[~observed, 'quote_signal'],
               color='#2463a6', s=16, alpha=.6, label='Filled quote')
    ax.scatter(frame.loc[observed, 'date'], frame.loc[observed, 'quote_signal'],
               color='#d46b25', s=55, zorder=5, label='Observed quote (preserved)')
    observed_dates = frame.loc[observed, 'date']
    ax.axvspan(observed_dates.min(), observed_dates.max(), color='#2463a6', alpha=.05)
    ax.set(title='December fixed route: linear quote signal fit',
           xlabel=f'Date; day = days since {fit["origin"]:%Y-%m-%d}; outside shaded interval = extrapolation',
           ylabel='Quote signal')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    return fig


def fill_shifted_market_index(frame: pd.DataFrame, curve_model: dict):
    """Fit only a constant vertical shift to an existing market equation.

    Least-squares shift = mean(observed market - base curve). Preserve observed
    values and use the shifted equation solely for missing dates. Repeated calls
    retain the original observation mask rather than fitting imputed values.
    """
    result = frame.copy()
    baseline = predict_market_curves(curve_model, result.date)
    observed = (result.market_index_observed.astype(bool).to_numpy()
                if 'market_index_observed' in result else result.market_index.notna().to_numpy())
    if not observed.any():
        raise ValueError('At least one observed market index is required to fit a shift')
    actual = result.loc[observed, 'market_index'].to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError('Observed market index must be finite')
    shift = float(np.mean(actual - baseline[observed]))
    shifted = baseline + shift
    if not np.isfinite(shifted).all() or (shifted <= 0).any():
        raise ValueError('Shifted market curve produces invalid indices')
    result['market_index_observed'] = observed
    result['market_index_base_curve'] = baseline
    result['market_index_shifted_curve'] = shifted
    result['market_index_source'] = np.where(observed, 'observed validation', 'base curve + constant shift')
    result.loc[~observed, 'market_index'] = shifted[~observed]
    fit = dict(vertical_shift=shift, observed_points=int(observed.sum()),
               filled_points=int((~observed).sum()),
               observed_fit_rmse=float(np.sqrt(np.mean((actual - shifted[observed]) ** 2))))
    return result, fit


def plot_shifted_market_index(frame: pd.DataFrame, fit: dict):
    """Compare the shared market equation, shifted curve and actual lane inputs."""
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    observed = frame.market_index_observed
    ax.plot(frame.date, frame.market_index_base_curve, color='#888888', lw=1, linestyle='--', label='Original market equation')
    ax.plot(frame.date, frame.market_index_shifted_curve, color='#2463a6', lw=1.5,
            label=f'Original equation {fit["vertical_shift"]:+.5f}')
    ax.scatter(frame.loc[~observed, 'date'], frame.loc[~observed, 'market_index'],
               color='#2463a6', s=16, label='Filled market index')
    ax.scatter(frame.loc[observed, 'date'], frame.loc[observed, 'market_index'],
               color='#d46b25', s=55, zorder=5, label='Observed market index (preserved)')
    ax.set(title='December fixed route: market curve + one fitted vertical shift',
           xlabel='Date', ylabel='Market index')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    return fig


def calibrate_rate_interval(actual, predicted, coverage: float = .90) -> dict:
    """Empirical actual/prediction quantiles; a transferred error band, not a coverage guarantee."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (actual.shape != predicted.shape or actual.ndim != 1 or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0).any() or (predicted <= 0).any() or not 0 < coverage < 1):
        raise ValueError('Interval calibration requires aligned positive finite rates and 0 < coverage < 1')
    ratios = actual / predicted
    lower, upper = np.quantile(ratios, [(1 - coverage) / 2, (1 + coverage) / 2])
    return dict(method='held-out empirical actual/predicted quantiles', nominal_coverage=coverage,
                lower_multiplier=float(lower), upper_multiplier=float(upper), calibration_rows=len(ratios),
                empirical_calibration_coverage=float(np.mean((ratios >= lower) & (ratios <= upper))),
                scope='Transferred model-error band; reconstructed-input uncertainty is excluded')


def apply_rate_interval(predictions: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    """Scale predictions by calibrated relative-error quantiles."""
    result = predictions.copy()
    lower, upper = calibration['lower_multiplier'], calibration['upper_multiplier']
    if not np.isfinite([lower, upper]).all() or not 0 < lower <= upper:
        raise ValueError('Invalid calibrated interval multipliers')
    result['lower_rate'] = result.predicted_rate * lower
    result['upper_rate'] = result.predicted_rate * upper
    return result


def plot_december_predictions(predictions: pd.DataFrame, calibration: dict):
    """Show fixed-route daily prices and an approximate model-error interval."""
    fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    ax.fill_between(predictions.date, predictions.lower_rate, predictions.upper_rate,
                    alpha=.18, color='#2463a6',
                    label=f'Approx. {100 * calibration["nominal_coverage"]:.0f}% model-error interval')
    ax.plot(predictions.date, predictions.predicted_rate, color='#2463a6', marker='o', ms=3,
            lw=1.5, label='Predicted rate')
    ax.set(title='December: Lexington → Fort Wayne | 360 miles | Dry Van | 32,000 lb',
           xlabel='Date; reconstructed-input uncertainty is not included', ylabel='Predicted rate')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    return fig


def compare_quote_product(cv_folds, cv_metrics: pd.DataFrame,
                          test: pd.DataFrame, test_predictions: pd.DataFrame):
    """Score the unadjusted quote_signal × distance baseline on identical held-out rows."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

    def metrics(actual, prediction):
        return dict(MAE=mean_absolute_error(actual, prediction),
                    RMSE=np.sqrt(mean_squared_error(actual, prediction)),
                    MAPE_pct=100 * mean_absolute_percentage_error(actual, prediction))

    rows = []
    for fold in cv_folds:
        baseline = fold['X_valid'].quote_signal * fold['X_valid'].distance
        row = cv_metrics.loc[cv_metrics.fold.eq(fold['fold'])].iloc[0]
        split = f'Fold {fold["fold"]}'
        rows.append(dict(split=split, model='CatBoost', **{key: row[key] for key in ['MAE', 'RMSE', 'MAPE_pct']}))
        rows.append(dict(split=split, model='Quote × distance', **metrics(fold['y_valid'], baseline)))
    if not test.index.equals(test_predictions.index) or not test.load_id.equals(test_predictions.load_id):
        raise ValueError('Test predictions must match baseline comparison rows')
    comparison = test[['load_id', 'date', 'posted_rate']].copy()
    comparison['CatBoost'] = test_predictions.predicted_rate
    comparison['Quote × distance'] = test.quote_signal * test.distance
    for name in ['CatBoost', 'Quote × distance']:
        rows.append(dict(split='Test', model=name, **metrics(comparison.posted_rate, comparison[name])))
    return pd.DataFrame(rows), comparison


def plot_quote_product_metrics(metrics: pd.DataFrame):
    """Grouped metric bars for CatBoost and a feature-product baseline per split."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    splits = metrics.split.drop_duplicates().tolist()
    positions = np.arange(len(splits))
    for ax, metric in zip(axes, ['MAE', 'RMSE', 'MAPE_pct']):
        for offset, name, color in [(-.19, 'CatBoost', '#2463a6'), (.19, 'Quote × distance', '#d46b25')]:
            values = metrics.loc[metrics.model.eq(name)].set_index('split').loc[splits, metric]
            bars = ax.bar(positions + offset, values, width=.36, label=name, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8, rotation=45)
        ax.set_xticks(positions, splits)
        ax.set(title=metric.replace('_pct', ' (%)'), ylabel='Percent' if metric == 'MAPE_pct' else 'Rate units')
        ax.set_ylim(0, metrics[metric].max() * 1.3)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same validation rows: CatBoost vs. quote signal × distance')
    return fig


def plot_quote_product_predictions(comparison: pd.DataFrame):
    """Compare test predictions with actual rates and daily mean actual prices."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for name, color in [('CatBoost', '#2463a6'), ('Quote × distance', '#d46b25')]:
        axes[0].scatter(comparison.posted_rate, comparison[name], s=8, alpha=.3, label=name, color=color)
    bounds = [comparison[['posted_rate', 'CatBoost', 'Quote × distance']].min().min(),
              comparison[['posted_rate', 'CatBoost', 'Quote × distance']].max().max()]
    axes[0].plot(bounds, bounds, color='black', lw=.8, linestyle='--', label='Ideal')
    axes[0].set(title='Held-out test: actual vs. predicted', xlabel='Actual rate',
                ylabel='Predicted rate', xscale='log', yscale='log')
    daily = comparison.groupby('date')[['posted_rate', 'CatBoost', 'Quote × distance']].mean()
    for column, label, color in [('posted_rate', 'Actual', '#333333'),
                                 ('CatBoost', 'CatBoost', '#2463a6'),
                                 ('Quote × distance', 'Quote × distance', '#d46b25')]:
        axes[1].plot(daily.index, daily[column], marker='o', ms=3, label=label, color=color)
    axes[1].set(title='Held-out test: daily mean rates', ylabel='Rate')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[1].tick_params(axis='x', rotation=30)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    return fig

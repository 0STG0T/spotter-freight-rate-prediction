"""Check residual semantics, SHAP paths and saved inference artifacts."""
from pathlib import Path
import json
import pickle
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from utils import test_residuals, worst_prediction_shap, save_model_bundle, calibrate_rate_interval, apply_rate_interval


class ModelArtifactTests(unittest.TestCase):
    def test_test_alignment_shap_and_reload(self):
        X = pd.DataFrame({'distance': np.arange(1., 21.)})
        y = 100 + X.distance * 10
        model = CatBoostRegressor(iterations=8, depth=2, verbose=False,
                                  thread_count=1, allow_writing_files=False)
        model.fit(X, np.log(y))
        actual = pd.DataFrame({'load_id': [f'id-{i}' for i in range(20)],
                               'date': pd.date_range('2025-01-01', periods=20), 'posted_rate': y})
        predictions = actual[['load_id']].assign(predicted_rate=np.exp(model.predict(X)))
        errors = test_residuals(actual, predictions)
        np.testing.assert_allclose(errors.residual, y - predictions.predicted_rate)
        with self.assertRaises(ValueError):
            test_residuals(actual, predictions.iloc[::-1])
        worst = errors.nlargest(5, 'APE_pct')
        fig, values = worst_prediction_shap(model, X, worst, [], max_display=1)
        np.testing.assert_allclose(values[['distance']].sum(axis=1) + values.base_value,
                                   values.predicted_log_rate)
        self.assertEqual(len(fig.axes[0].lines), 5)
        plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            save_model_bundle(model, tmp, feature_state={'median': 10.}, feature_columns=X.columns,
                              cat_features=[], city_centers=pd.DataFrame(), route_table=pd.DataFrame(),
                              metadata={'target_transform': 'log'}, verification_features=X)
            state = pickle.loads((Path(tmp) / 'preprocessing.pkl').read_bytes())
            self.assertEqual(state['feature_columns'], ['distance'])
            self.assertEqual(state['feature_state'], {'median': 10.})
            self.assertEqual(json.loads((Path(tmp) / 'metadata.json').read_text())['target_transform'], 'log')
            restored = CatBoostRegressor().load_model(str(Path(tmp) / 'model.cbm'))
            np.testing.assert_allclose(model.predict(X), restored.predict(X))

    def test_relative_error_interval_scales_and_rejects_invalid_inputs(self):
        predicted = np.array([100., 200., 300., 400.])
        actual = predicted * np.array([.8, .9, 1.1, 1.2])
        calibration = calibrate_rate_interval(actual, predicted, coverage=.5)
        scaled = calibrate_rate_interval(actual * 10, predicted * 10, coverage=.5)
        self.assertEqual(calibration, scaled)
        frame = pd.DataFrame({'predicted_rate': [100., 200.]})
        result = apply_rate_interval(frame, calibration)
        np.testing.assert_allclose(result.lower_rate, frame.predicted_rate * .875)
        np.testing.assert_allclose(result.upper_rate, frame.predicted_rate * 1.125)
        with self.assertRaises(ValueError):
            calibrate_rate_interval([1., 2.], [0., 1.])


if __name__ == '__main__':
    unittest.main()

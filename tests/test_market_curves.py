"""Regression checks for stable market extrapolation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from utils import fit_market_curves, predict_market_curves, weekly_basis, fill_shifted_market_index


class MarketCurveTests(unittest.TestCase):
    def test_same_equation_inside_and_outside_training(self):
        origin = pd.Timestamp('2025-01-01')
        coefficients = np.array([.9, -.1, .02, .035, -.083])
        model = {'origin': origin, 'segments': [
            {'start': origin, 'end': pd.Timestamp('2025-10-17'), 'coefficients': coefficients}]}
        dates = pd.date_range('2025-09-27', '2025-12-31')
        t = (dates - origin).days.to_numpy() / 365.25
        expected = np.column_stack([np.ones(len(t)), t, t*t, weekly_basis(dates)]) @ coefficients
        np.testing.assert_allclose(predict_market_curves(model, dates), expected)

    def test_short_final_regime_does_not_absorb_previous_days(self):
        dates = pd.date_range('2025-01-01', periods=90)
        t = np.arange(90) / 365.25
        values = .9 + .2*t + .03*t*t + weekly_basis(dates) @ np.array([.05, -.08])
        values += np.select([np.arange(90) < 20, np.arange(90) < 40,
                             np.arange(90) < 75], [0., .4, -.3], default=.1)
        model = fit_market_curves(pd.DataFrame({'date': dates, 'market_index': values}), n_segments=4)
        self.assertEqual(model['segments'][-1]['start'], dates[75])
        np.testing.assert_allclose(predict_market_curves(model, dates), values, atol=1e-8)

    def test_shift_changes_only_intercept_and_preserves_observations(self):
        dates = pd.date_range('2025-12-01', periods=7)
        model = {'origin': dates[0], 'segments': [
            {'start': dates[0], 'end': dates[-1], 'coefficients': np.array([1., .1, .02, .03, .04])}]}
        base = predict_market_curves(model, dates)
        observed = np.array([0, 3, 6])
        frame = pd.DataFrame({'date': dates, 'market_index': np.nan})
        frame.loc[observed, 'market_index'] = base[observed] + .1 + np.array([.01, -.01, 0.])
        filled, fit = fill_shifted_market_index(frame, model)
        self.assertAlmostEqual(fit['vertical_shift'], .1)
        np.testing.assert_allclose(filled.market_index_shifted_curve - base, .1)
        pd.testing.assert_series_equal(filled.loc[observed, 'market_index'], frame.loc[observed, 'market_index'])
        self.assertTrue(filled.market_index.notna().all())
        repeated, fit2 = fill_shifted_market_index(filled, model)
        pd.testing.assert_frame_equal(filled, repeated)
        self.assertEqual(fit, fit2)


if __name__ == '__main__':
    unittest.main()

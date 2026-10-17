import numpy as np

from flare_lag.discover import best_lag, lagged_correlations, same_day_blame
from flare_lag.predict import design_matrix
from flare_lag.simulate import PLANTED_LAGS, simulate


def test_lag_scan_recovers_planted_delays():
    diary = simulate(n_people=250, n_days=100, seed=3, missing_rate=0.0)
    correlations = lagged_correlations(diary)
    for name, planted in PLANTED_LAGS.items():
        if planted is None:
            continue
        lag, r = best_lag(correlations, name)
        assert lag == planted
        assert r > 0.05


def test_same_day_look_blames_the_cream():
    diary = simulate(n_people=250, n_days=100, seed=3, missing_rate=0.0)
    blamed = same_day_blame(lagged_correlations(diary))
    assert blamed == "cream"


def test_features_do_not_read_the_future():
    diary = simulate(n_people=4, n_days=12, seed=1, missing_rate=0.0)
    diary.stress[:, :] = 0
    diary.stress[0, 5] = 9
    x, _y, person, names = design_matrix(diary, (2,))
    col = names.index("stress_lag2")
    # Day index in the design starts at lag 2, so row for person 0 day 7 uses stress on day 5.
    row = np.where(person == 0)[0][5]
    assert x[row, col] == 9
    # The flare day itself is not inside a lag-2 feature for that same row's later day.
    later = np.where(person == 0)[0][4]
    assert x[later, col] == 0

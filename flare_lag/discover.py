"""Find which signal lines up with flares, and how many days earlier."""

from __future__ import annotations

import numpy as np

from flare_lag.simulate import CAUSES, DECOYS, SIGNALS, Diary


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan
    xs = x[mask]
    ys = y[mask].astype(float)
    if xs.std() < 1e-8 or ys.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def lagged_correlations(diary: Diary, max_lag: int = 4) -> dict[tuple[str, int], float]:
    """Pooled correlation of signal[t - lag] with flare[t], across people."""
    flare = diary.flare
    out: dict[tuple[str, int], float] = {}
    for name in SIGNALS:
        signal = diary.signal(name)
        for lag in range(max_lag + 1):
            if lag == 0:
                x = signal.ravel()
                y = flare.ravel()
            else:
                x = signal[:, :-lag].ravel()
                y = flare[:, lag:].ravel()
            out[(name, lag)] = _corr(x, y)
    return out


def best_lag(correlations: dict[tuple[str, int], float], signal: str) -> tuple[int, float]:
    pairs = [(lag, r) for (name, lag), r in correlations.items() if name == signal and np.isfinite(r)]
    if not pairs:
        return -1, np.nan
    lag, r = max(pairs, key=lambda item: abs(item[1]))
    return lag, r


def same_day_blame(correlations: dict[tuple[str, int], float]) -> str:
    """The signal a same-day look would blame. Cream is included."""
    pairs = [(name, r) for (name, lag), r in correlations.items() if lag == 0 and np.isfinite(r)]
    name, _r = max(pairs, key=lambda item: abs(item[1]))
    return name


def recovery(correlations: dict[tuple[str, int], float]) -> dict[str, dict[str, float | int | bool | str]]:
    """Did the strongest lag for each cause match the planted delay?"""
    from flare_lag.simulate import PLANTED_LAGS

    report: dict[str, dict[str, float | int | bool | str]] = {}
    for name in CAUSES:
        lag, r = best_lag(correlations, name)
        planted = int(PLANTED_LAGS[name])  # type: ignore[arg-type]
        report[name] = {
            "planted_lag": planted,
            "best_lag": lag,
            "correlation": round(r, 3),
            "recovered": lag == planted,
        }
    decoy_flags = {}
    for name in DECOYS:
        lag, r = best_lag(correlations, name)
        decoy_flags[name] = {"best_lag": lag, "correlation": round(float(r), 3)}
    blamed = same_day_blame(correlations)
    return {
        "causes": report,  # type: ignore[dict-item]
        "decoys": decoy_flags,  # type: ignore[dict-item]
        "same_day_blame": blamed,
        "same_day_blames_cream": blamed == "cream",
    }

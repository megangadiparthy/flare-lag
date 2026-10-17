"""Predict today's flare from earlier days, without using the cream put on after it."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flare_lag.simulate import Diary

PREDICTORS = ("stress", "short_sleep", "sugar", "steps", "screens")


def _impute(values: np.ndarray) -> np.ndarray:
    filled = values.astype(float).copy()
    person_mean = np.nanmean(filled, axis=1)
    person_mean = np.where(np.isfinite(person_mean), person_mean, 0.0)
    inds = np.where(~np.isfinite(filled))
    filled[inds] = person_mean[inds[0]]
    return filled


def design_matrix(diary: Diary, lags: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Rows are person-days. Features use only signal[t - lag], so lag 0 is the same day."""
    filled = {name: _impute(diary.signal(name)) for name in PREDICTORS}
    n_people, n_days = diary.flare.shape
    start = max(lags) if lags else 0
    rows: list[np.ndarray] = []
    y: list[np.ndarray] = []
    people: list[np.ndarray] = []
    names: list[str] = []
    columns: list[np.ndarray] = []
    for name in PREDICTORS:
        for lag in lags:
            names.append(f"{name}_lag{lag}")
            if lag == 0:
                columns.append(filled[name][:, start:].ravel())
            else:
                columns.append(filled[name][:, start - lag : n_days - lag].ravel())
    x = np.column_stack(columns) if columns else np.zeros((n_people * (n_days - start), 0))
    target = diary.flare[:, start:].ravel().astype(int)
    person = np.repeat(np.arange(n_people), n_days - start)
    rows.append(x)
    y.append(target)
    people.append(person)
    return rows[0], y[0], people[0], names


def _fit_auc(x_train, y_train, x_test, y_test) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "flare_rate_test": float(y_test.mean()),
    }


def person_split(person: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.unique(person)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test_ids = set(ids[:n_test].tolist())
    test = np.array([p in test_ids for p in person])
    return ~test, test


def compare_models(diary: Diary, seed: int = 7) -> dict[str, dict[str, float]]:
    """Same-day features versus the one-to-three-day window, split by person."""
    specs = {
        "same_day": (0,),
        "lagged_1_to_3": (1, 2, 3),
    }
    out: dict[str, dict[str, float]] = {}
    for label, lags in specs.items():
        x, y, person, _names = design_matrix(diary, lags)
        train, test = person_split(person, test_fraction=0.3, seed=seed)
        metrics = _fit_auc(x[train], y[train], x[test], y[test])
        out[label] = {key: round(value, 3) for key, value in metrics.items()}
    return out


def coefficient_ranks(diary: Diary) -> list[dict[str, float | str]]:
    """Fit the lagged model on everyone and rank absolute coefficients. Descriptive only."""
    x, y, _person, names = design_matrix(diary, (1, 2, 3))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, class_weight="balanced"),
    )
    model.fit(x, y)
    coefs = model.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    return [
        {"feature": names[i], "coefficient": round(float(coefs[i]), 3)}
        for i in order[:8]
    ]

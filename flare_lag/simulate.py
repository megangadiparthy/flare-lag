"""Synthetic daily diaries with a known delay between a habit and a flare.

The diaries are not patient records. Each person has three planted triggers
and three decoys. Cream is applied because a flare is already happening, so
it lines up with the same day and is not a cause.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# signal name -> day offset before the flare. None means the signal is a decoy.
PLANTED_LAGS: dict[str, int | None] = {
    "stress": 2,
    "short_sleep": 1,
    "sugar": 3,
    "steps": None,
    "screens": None,
    "cream": None,
}

CAUSES = ("stress", "short_sleep", "sugar")
DECOYS = ("steps", "screens", "cream")
SIGNALS = tuple(PLANTED_LAGS)


@dataclass(frozen=True)
class Diary:
    """One row per person-day. Arrays are shaped (people, days)."""

    stress: np.ndarray
    short_sleep: np.ndarray
    sugar: np.ndarray
    steps: np.ndarray
    screens: np.ndarray
    cream: np.ndarray
    flare: np.ndarray
    person: np.ndarray
    day: np.ndarray

    def signal(self, name: str) -> np.ndarray:
        return getattr(self, name)


def _ar(rng: np.random.Generator, n_people: int, n_days: int, phi: float, sigma: float) -> np.ndarray:
    out = np.zeros((n_people, n_days))
    out[:, 0] = rng.normal(0.0, sigma, n_people)
    noise = rng.normal(0.0, sigma, (n_people, n_days - 1))
    for t in range(1, n_days):
        out[:, t] = phi * out[:, t - 1] + noise[:, t - 1]
    return out


def simulate(
    n_people: int = 400,
    n_days: int = 120,
    seed: int = 7,
    missing_rate: float = 0.08,
) -> Diary:
    """Build diaries. Effects are strong enough to recover, with carryover and noise."""
    rng = np.random.default_rng(seed)
    stress = _ar(rng, n_people, n_days, phi=0.55, sigma=1.0)
    sleep_hours = 7.2 + _ar(rng, n_people, n_days, phi=0.4, sigma=0.75)
    short_sleep = np.clip((6.5 - sleep_hours) / 1.4, 0.0, 3.0)
    sugar = np.clip(1.1 + _ar(rng, n_people, n_days, phi=0.35, sigma=0.85), 0.0, None)
    steps = _ar(rng, n_people, n_days, phi=0.3, sigma=1.0)
    screens = _ar(rng, n_people, n_days, phi=0.45, sigma=1.0)

    flare = np.zeros((n_people, n_days), dtype=np.int8)
    for t in range(n_days):
        logit = np.full(n_people, -1.55)
        if t >= 2:
            logit = logit + 0.9 * stress[:, t - 2]
        if t >= 1:
            logit = logit + 0.85 * short_sleep[:, t - 1]
        if t >= 3:
            logit = logit + 0.7 * sugar[:, t - 3]
        if t >= 1:
            logit = logit + 1.05 * flare[:, t - 1]
        logit = logit + rng.normal(0.0, 0.45, n_people)
        flare[:, t] = rng.random(n_people) < (1.0 / (1.0 + np.exp(-logit)))

    cream = ((flare == 1) & (rng.random((n_people, n_days)) < 0.72)).astype(float)
    cream = np.clip(cream + ((rng.random((n_people, n_days)) < 0.05) & (flare == 0)), 0.0, 1.0)

    observed = {
        "stress": stress.copy(),
        "short_sleep": short_sleep.copy(),
        "sugar": sugar.copy(),
        "steps": steps.copy(),
        "screens": screens.copy(),
    }
    for values in observed.values():
        drop = rng.random(values.shape) < missing_rate
        values[drop] = np.nan

    person = np.repeat(np.arange(n_people), n_days)
    day = np.tile(np.arange(n_days), n_people)
    return Diary(
        stress=observed["stress"],
        short_sleep=observed["short_sleep"],
        sugar=observed["sugar"],
        steps=observed["steps"],
        screens=observed["screens"],
        cream=cream,
        flare=flare,
        person=person,
        day=day,
    )

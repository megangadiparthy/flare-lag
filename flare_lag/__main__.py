"""Run the study and write results/summary.json plus one figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare_lag.discover import lagged_correlations, recovery
from flare_lag.predict import coefficient_ranks, compare_models
from flare_lag.simulate import CAUSES, PLANTED_LAGS, simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _figure(found: dict, models: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    labels = []
    planted = []
    chosen = []
    for name in CAUSES:
        row = found["causes"][name]
        labels.append(name.replace("_", " "))
        planted.append(row["planted_lag"])
        chosen.append(row["best_lag"])
    x = np.arange(len(labels))
    axes[0].bar(x - 0.16, planted, width=0.32, color="#C4607A", label="Planted delay")
    axes[0].bar(x + 0.16, chosen, width=0.32, color="#221D27", label="Strongest lag")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Days before the flare")
    axes[0].set_ylim(0, 4.5)
    axes[0].set_title("Did the scan find the planted delay?")
    axes[0].legend(frameon=False, fontsize=8)

    names = ["Same day", "1–3 days earlier"]
    aucs = [models["same_day"]["auc"], models["lagged_1_to_3"]["auc"]]
    axes[1].bar(names, aucs, color=["#777267", "#C4607A"])
    axes[1].axhline(0.5, color="#CBC6B9", lw=1)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_ylabel("Test AUC")
    axes[1].set_title("Predicting a flare on held-out people")
    for i, value in enumerate(aucs):
        axes[1].text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    diary = simulate()
    correlations = lagged_correlations(diary)
    found = recovery(correlations)
    models = compare_models(diary)
    ranks = coefficient_ranks(diary)
    flare_rate = float(diary.flare.mean())
    top5_causes = sum(
        1 for row in ranks[:5] if any(str(row["feature"]).startswith(name) for name in CAUSES)
    )

    summary = {
        "people": int(diary.flare.shape[0]),
        "days": int(diary.flare.shape[1]),
        "flare_days": round(flare_rate, 3),
        "planted_lags": {name: PLANTED_LAGS[name] for name in CAUSES},
        "discovery": found,
        "prediction": models,
        "top_coefficients": ranks,
        "top5_planted_cause_count": top5_causes,
        "note": (
            "Diaries are simulated. Cream is applied because a flare is already happening. "
            "This does not diagnose anyone and does not use real health records."
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _figure(found, models, RESULTS / "lag_vs_sameday.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

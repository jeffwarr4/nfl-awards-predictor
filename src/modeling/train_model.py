"""
Train the NFL MVP model(s) from data/processed/training_data.csv.

Targets (config.TARGETS; config.MODEL_TARGET picks the one predict uses)
- vote_share: label = received any MVP vote; vote-getters weighted 1 + 20*share
              StandardScaler + LogisticRegression(C=0.5, class_weight="balanced")
- winner:     label = won MVP (the 2025 model)
              StandardScaler + LogisticRegression(class_weight="balanced")

Evaluation
- Leave-one-season-out over TRAIN_START..TRAIN_END
- Each held-out season is ranked on a QB/RB board; we record where the actual
  winner landed (best-ranked winner in co-MVP years) and report winner #1 /
  top-3 / top-5 rates and mean rank. AUC is not reported — with ~1 positive per
  season it says nothing about whether the winner is at the top of the board.

Outputs (per target)
- models/nfl_mvp_{target}.joblib
- models/metrics_{target}.json        (features, per-season winner rank, summary)
- models/coefficients_{target}.csv

Run:
    python -m src.modeling.train_model                    # both targets
    python -m src.modeling.train_model --target winner
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    TRAINING_CSV, MODELS_DIR, FEATURES, TARGETS, MODEL_TARGET, SHARE_WEIGHT,
    TRAIN_START, TRAIN_END, BOARD_POSITIONS, model_path, metrics_path,
)
from src.features import add_derived_features, feature_matrix


def build_model(target: str) -> Pipeline:
    if target == "vote_share":
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000)
    elif target == "winner":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs")
    else:
        raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def labels_and_weights(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "vote_share":
        y = df["any_votes"].astype(int).to_numpy()
        w = np.where(y == 1, 1.0 + SHARE_WEIGHT * df["vote_share"].to_numpy(), 1.0)
    else:
        y = df["mvp_winner"].astype(int).to_numpy()
        w = np.ones(len(df))
    return y, w


def fit(df: pd.DataFrame, target: str) -> Pipeline:
    y, w = labels_and_weights(df, target)
    model = build_model(target)
    model.fit(feature_matrix(df, FEATURES), y, clf__sample_weight=w)
    return model


def winner_rank(board: pd.DataFrame, prob: np.ndarray) -> int | None:
    """Rank (1 = top) of the best-ranked MVP winner on a season's board."""
    order = pd.Series(prob, index=board.index).rank(ascending=False, method="min")
    winners = order[board["mvp_winner"] == 1]
    return int(winners.min()) if len(winners) else None


def backtest(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    seasons = sorted(df["season"].unique())
    per_season = []
    for season in seasons:
        train = df[df["season"] != season]
        test = df[(df["season"] == season) & df["position"].isin(BOARD_POSITIONS)]
        model = fit(train, target)
        prob = model.predict_proba(feature_matrix(test, FEATURES))[:, 1]
        rank = winner_rank(test, prob)
        top = test.assign(p=prob).sort_values("p", ascending=False).iloc[0]
        winner = ", ".join(test.loc[test["mvp_winner"] == 1, "player_display_name"])
        per_season.append({"season": int(season), "winner": winner,
                           "winner_rank": rank, "model_top1": top["player_display_name"]})

    ranks = [r["winner_rank"] for r in per_season if r["winner_rank"] is not None]
    summary = {
        "seasons": len(ranks),
        "winner_top1": float(np.mean([r <= 1 for r in ranks])),
        "winner_top3": float(np.mean([r <= 3 for r in ranks])),
        "winner_top5": float(np.mean([r <= 5 for r in ranks])),
        "winner_mean_rank": float(np.mean(ranks)),
    }
    return {"per_season": per_season, "summary": summary}


def run(target: str) -> Dict[str, Any]:
    df = pd.read_csv(TRAINING_CSV)
    df = df[df["season"].between(TRAIN_START, TRAIN_END)]
    df = add_derived_features(df)

    bt = backtest(df, target)

    model = fit(df, target)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, model_path(target))

    coefs = pd.DataFrame({"feature": FEATURES,
                          "coef": model.named_steps["clf"].coef_[0]})
    coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index) \
         .to_csv(MODELS_DIR / f"coefficients_{target}.csv", index=False)

    metrics = {
        "target": target,
        "train_seasons": [TRAIN_START, TRAIN_END],
        "n_samples": int(len(df)),
        "features_used": FEATURES,
        "backtest": bt,
    }
    with open(metrics_path(target), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=[*TARGETS, "both"], default="both")
    args = parser.parse_args()
    targets = TARGETS if args.target == "both" else (args.target,)

    results = {t: run(t) for t in targets}

    print(f"Leave-one-season-out, {TRAIN_START}-{TRAIN_END}, QB/RB board "
          f"(live model: {MODEL_TARGET})")
    print(f"{'target':<12} {'#1':>6} {'top3':>6} {'top5':>6} {'mean rank':>10}")
    for t, m in results.items():
        s = m["backtest"]["summary"]
        print(f"{t:<12} {s['winner_top1']:>6.0%} {s['winner_top3']:>6.0%} "
              f"{s['winner_top5']:>6.0%} {s['winner_mean_rank']:>10.1f}")

    print("\nWinner rank by season:")
    seasons = results[targets[0]]["backtest"]["per_season"]
    header = "  ".join(f"{t:>10}" for t in targets)
    print(f"  {'season':<6} {'winner':<34} {header}")
    for i, row in enumerate(seasons):
        ranks = "  ".join(f"{str(results[t]['backtest']['per_season'][i]['winner_rank']):>10}"
                          for t in targets)
        print(f"  {row['season']:<6} {row['winner']:<34} {ranks}")


if __name__ == "__main__":
    main()

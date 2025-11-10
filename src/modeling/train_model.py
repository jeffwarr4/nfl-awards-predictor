"""
Train baseline NFL MVP models from data/processed/training_data.csv
with engineered features that strengthen team-success and sample-size signals.

Models
- LogisticRegression (scaled, class_weight='balanced')
- HistGradientBoostingClassifier (class_weight='balanced')

Evaluation
- Grouped CV by season (no leakage)
- ROC AUC
- Top-1 / Top-3 in-season hit rates

Outputs
- models/nfl_mvp_logreg.pkl
- models/nfl_mvp_hgb.pkl
- outputs/training_metrics.json
- outputs/feature_importance_hgb.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from joblib import dump

# ------------------------
# Paths
# ------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "training_data.csv"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"

SEASON_COL = "season"
TARGET_COL = "mvp_winner"
NAME_COL = "player_display_name"  # falls back later if missing

# Base features (no fantasy_points)
BASE_FEATURES: List[str] = [
    "games", "attempts", "completions",
    "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
    "rushing_yards", "rushing_tds", "rushing_epa",
    "receiving_yards", "receiving_tds", "receiving_epa",
    "pass_ypg", "rush_ypg", "recv_ypg",
    "win_pct",
    "td_int_ratio",
    "is_qb", "is_rb", "is_wr",
]

# ------------------------
# Utilities
# ------------------------
def _available_features(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    cols = [c for c in candidates if c in df.columns]
    cols = [c for c in cols if not df[c].isna().all()]
    return cols

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TD/INT ratio (robust)
    if {"passing_tds", "passing_interceptions"}.issubset(df.columns):
        denom = df["passing_interceptions"].replace(0, np.nan)
        df["td_int_ratio"] = (df["passing_tds"] / denom).fillna(df["passing_tds"]).astype(float)
    else:
        df["td_int_ratio"] = 0.0

    # Position flags (if present)
    if "position" in df.columns:
        df["is_qb"] = (df["position"] == "QB").astype(int)
        df["is_rb"] = (df["position"] == "RB").astype(int)
        df["is_wr"] = (df["position"] == "WR").astype(int)
    else:
        df["is_qb"] = df["is_rb"] = df["is_wr"] = 0

    # --- New engineered features ---
    # Stronger team-success signal and interactions
    if "win_pct" not in df.columns:
        df["win_pct"] = 0.0
    df["win_pct"] = df["win_pct"].astype(float).fillna(0.0)

    if "games" not in df.columns:
        df["games"] = 0
    df["games"] = pd.to_numeric(df["games"], errors="coerce").fillna(0).astype(float)

    df["win_pct_scaled"] = df["win_pct"] * 10.0
    df["games_weighted_win"] = df["games"] * df["win_pct"]
    df["efficiency_win"] = df["td_int_ratio"] * df["win_pct_scaled"]

    # Optional sample-size realism (kept for reference)
    df["games_penalty"] = np.where(df["games"] < 8, 0.5, 1.0).astype(float)

    return df

def _season_topk_hit_rate(df_pred: pd.DataFrame, prob_col: str, k: int = 1) -> float:
    hits = []
    for season, grp in df_pred.groupby(SEASON_COL):
        if grp[TARGET_COL].sum() == 0:
            continue
        topk = grp.sort_values(prob_col, ascending=False).head(k)
        hits.append(int(topk[TARGET_COL].max() == 1))
    return float(np.mean(hits)) if hits else 0.0

# ------------------------
# Training
# ------------------------
def run(
    data_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    outputs_dir: Optional[Union[str, Path]] = None,
) -> str:
    data_path = Path(data_path) if data_path else DATA_PATH
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    outputs_dir = Path(outputs_dir) if outputs_dir else OUTPUTS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # Required columns
    req = {SEASON_COL, TARGET_COL}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Feature engineering
    df = _add_derived_features(df)

    # Final feature list (base + engineered)
    FEATURE_SET = BASE_FEATURES + ["win_pct_scaled", "games_weighted_win", "efficiency_win"]
    features = _available_features(df, FEATURE_SET)

    # Target/group
    y = df[TARGET_COL].astype("int32").to_numpy()
    groups = pd.to_numeric(df[SEASON_COL], errors="coerce").fillna(0).astype("int32").to_numpy()


    # Feature matrix (fillna)
    X = df[features].copy().fillna(0.0)

    # Pipelines
    num_transformer = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=True, with_std=True), list(range(X.shape[1])))],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logreg = Pipeline(
        steps=[
            ("scale", num_transformer),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=3,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=42,
    )

    # Grouped CV by season
    n_splits = min(5, df[SEASON_COL].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    metrics: Dict[str, Any] = {"features_used": features, "cv": []}
    oof_logreg = np.zeros(len(df))
    oof_hgb = np.zeros(len(df))

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        logreg.fit(Xtr, ytr)
        pr_log = logreg.predict_proba(Xte)[:, 1]; oof_logreg[te] = pr_log

        hgb.fit(Xtr, ytr)
        pr_hgb = hgb.predict_proba(Xte)[:, 1]; oof_hgb[te] = pr_hgb

        fold_df = df.iloc[te].copy()
        fold_df["pr_logreg"] = pr_log
        fold_df["pr_hgb"] = pr_hgb

        auc_log = roc_auc_score(yte, pr_log) if len(np.unique(yte)) > 1 else float("nan")
        auc_hgb = roc_auc_score(yte, pr_hgb) if len(np.unique(yte)) > 1 else float("nan")
        top1_log = _season_topk_hit_rate(fold_df, "pr_logreg", k=1)
        top1_hgb = _season_topk_hit_rate(fold_df, "pr_hgb", k=1)
        top3_log = _season_topk_hit_rate(fold_df, "pr_logreg", k=3)
        top3_hgb = _season_topk_hit_rate(fold_df, "pr_hgb", k=3)

        metrics["cv"].append({
            "fold": fold,
            "seasons": sorted(set(fold_df[SEASON_COL].tolist())),
            "auc_logreg": auc_log,
            "auc_hgb": auc_hgb,
            "top1_logreg": top1_log,
            "top1_hgb": top1_hgb,
            "top3_logreg": top3_log,
            "top3_hgb": top3_hgb,
        })

    # Overall metrics
    df_eval = df.copy()
    df_eval["pr_logreg"] = oof_logreg
    df_eval["pr_hgb"] = oof_hgb

    metrics["overall"] = {
        "auc_logreg": float(roc_auc_score(y, oof_logreg)),
        "auc_hgb": float(roc_auc_score(y, oof_hgb)),
        "top1_logreg": _season_topk_hit_rate(df_eval, "pr_logreg", k=1),
        "top1_hgb": _season_topk_hit_rate(df_eval, "pr_hgb", k=1),
        "top3_logreg": _season_topk_hit_rate(df_eval, "pr_logreg", k=3),
        "top3_hgb": _season_topk_hit_rate(df_eval, "pr_hgb", k=3),
        "n_samples": int(len(df)),
        "n_seasons": int(df[SEASON_COL].nunique()),
        "pos_rate": float(df[TARGET_COL].mean()),
    }

    # Fit on all data and persist
    logreg.fit(X, y); hgb.fit(X, y)
    dump(logreg, models_dir / "nfl_mvp_logreg.pkl")
    dump(hgb, models_dir / "nfl_mvp_hgb.pkl")

    # Permutation importance for HGB
    try:
        pi = permutation_importance(
            hgb, X, y, n_repeats=10, random_state=42, scoring="roc_auc"
        )

        # Handle both dict-like and Bunch return types
        importances_mean = getattr(pi, "importances_mean", pi["importances_mean"])
        importances_std = getattr(pi, "importances_std", pi["importances_std"])

        fi = (
            pd.DataFrame({
                "feature": features,
                "importance_mean": importances_mean,
                "importance_std": importances_std,
            })
            .sort_values("importance_mean", ascending=False)
        )
        fi.to_csv(outputs_dir / "feature_importance_hgb.csv", index=False)
    except Exception as e:
        with open(outputs_dir / "feature_importance_hgb.csv", "w", encoding="utf-8") as f:
            f.write(f"# permutation importance failed: {e}\n")


    with open(outputs_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    overall = metrics["overall"]
    print(
        "Training complete\n"
        f"- Features used: {len(features)}\n"
        f"- Overall AUC:   logreg={overall['auc_logreg']:.3f} | hgb={overall['auc_hgb']:.3f}\n"
        f"- Top-1 hit %:   logreg={overall['top1_logreg']:.3f} | hgb={overall['top1_hgb']:.3f}\n"
        f"- Top-3 hit %:   logreg={overall['top3_logreg']:.3f} | hgb={overall['top3_hgb']:.3f}\n"
        f"- Seasons: {overall['n_seasons']} | Samples: {overall['n_samples']} | PosRate: {overall['pos_rate']:.4f}"
    )

    return str(models_dir)

if __name__ == "__main__":
    run()
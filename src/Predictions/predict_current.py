"""
Predict current-season NFL MVP probabilities using the trained Logistic Regression model.

- Loads weekly player stats and aggregates to SEASON TOTALS
- Computes team win% from played games only
- Engineers features to match training (win_pct_scaled, games_weighted_win, efficiency_win)
- Aligns features exactly to training_metrics.json -> features_used
- Saves ALL, Top10, Top5 CSVs; includes headshot_url (if present) and local team_logo_path

Run:
    python -m src.Predictions.predict_current
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load
import nflreadpy as nfl

# GitHub base for hosted assets (used for Canva image URLs)
GITHUB_USER = "jeffwarr4"              # 👈 update with your username
GITHUB_REPO = "nfl-awards-predictor"
GITHUB_BRANCH = "main"

CDN_BASE = f"https://jeffwarr4.github.io/nfl-awards-predictor"



# ------------------------
# Config
# ------------------------
SEASON = 2025  # update each year

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
#ASSETS_DIR = REPO_ROOT / "assets" / "logos"  # put logos as {TEAM}.png (e.g., KC.png)

MODEL_PATH = MODELS_DIR / "nfl_mvp_logreg.pkl"  # we use Logistic Regression
TRAINING_METRICS_PATH = OUTPUTS_DIR / "training_metrics.json"

OUT_ALL   = OUTPUTS_DIR / f"mvp_predictions_{SEASON}_all.csv"
OUT_TOP10 = OUTPUTS_DIR / f"mvp_predictions_{SEASON}_top10.csv"
OUT_TOP5  = OUTPUTS_DIR / f"mvp_predictions_{SEASON}_top5.csv"


# ------------------------
# Helpers
# ------------------------
def _clean_name(s_like) -> pd.Series:
    import re
    s = pd.Series(s_like, dtype="string")
    suffix_re = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.IGNORECASE)
    punct_re = re.compile(r"[^\w\s]")
    def norm(x: str) -> str:
        x = str(x).strip().lower()
        x = punct_re.sub(" ", x)
        tokens = [t for t in x.split() if len(t) > 1 or not t.isalpha()]
        x = " ".join(tokens)
        x = suffix_re.sub("", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x
    return s.map(norm)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TD:INT ratio (robust to INT=0)
    if {"passing_tds", "passing_interceptions"}.issubset(df.columns):
        denom = pd.to_numeric(df["passing_interceptions"], errors="coerce").replace(0, np.nan)
        df["td_int_ratio"] = (pd.to_numeric(df["passing_tds"], errors="coerce") / denom).fillna(df["passing_tds"]).astype(float)
    else:
        df["td_int_ratio"] = 0.0

    # Position flags
    if "position" in df.columns:
        df["is_qb"] = (df["position"] == "QB").astype(int)
        df["is_rb"] = (df["position"] == "RB").astype(int)
        df["is_wr"] = (df["position"] == "WR").astype(int)
    else:
        df["is_qb"] = df["is_rb"] = df["is_wr"] = 0

    # Ensure games/win_pct exist as Series, then numeric/safe
    if "games" not in df.columns:
        df["games"] = 0
    df["games"] = pd.to_numeric(df["games"], errors="coerce").fillna(0).astype(float)

    if "win_pct" not in df.columns:
        df["win_pct"] = 0.0
    df["win_pct"] = pd.to_numeric(df["win_pct"], errors="coerce").fillna(0.0).clip(0, 1)

    # Per-game metrics (safe divide)
    games_nonzero = df["games"].replace(0, np.nan)
    for src, tgt in [("passing_yards", "pass_ypg"),
                     ("rushing_yards", "rush_ypg"),
                     ("receiving_yards", "recv_ypg")]:
        if src in df.columns:
            df[tgt] = (pd.to_numeric(df[src], errors="coerce") / games_nonzero).fillna(0.0)
        else:
            df[tgt] = 0.0

    # Engineered features (match training)
    df["win_pct_scaled"]    = df["win_pct"] * 10.0
    df["games_weighted_win"] = df["games"] * df["win_pct"]
    df["efficiency_win"]     = df["td_int_ratio"] * df["win_pct_scaled"]

    return df


def _aggregate_to_season(players_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly rows to one row per player-season-team with sums.
    """
    group_keys = ["season", "player_id", "player_display_name", "position", "team"]

    # Sum only numeric columns, EXCLUDING group keys and 'week'
    numeric_cols = [
        c for c in players_weekly.columns
        if c not in group_keys + ["week"]
           and pd.api.types.is_numeric_dtype(players_weekly[c])
    ]

    agg = (
        players_weekly
        .groupby(group_keys, dropna=False)[numeric_cols]
        .sum()
        .reset_index()
    )

    # Derive games if not present or all zeros: number of distinct weeks per player-season-team
    if "games" not in agg.columns or agg["games"].eq(0).all():
        wks = (
            players_weekly
            .groupby(group_keys, dropna=False)["week"]
            .nunique()
            .rename("games")
            .reset_index()
        )
        agg = agg.merge(wks, on=group_keys, how="left")

    return agg


# ------------------------
# Main
# ------------------------
def run(season: int = SEASON) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    #ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {season} weekly player stats and schedules...")

    # 1) Load weekly player stats (nflreadpy -> pandas) and aggregate to season totals
    players_pl = nfl.load_player_stats([season])  # weekly rows
    players_w = players_pl.to_pandas()
    players = _aggregate_to_season(players_w)

    # <-- ADD THIS: carry a single headshot per player_id
    if "headshot_url" in players_w.columns:
        headshots = (
            players_w[["player_id", "headshot_url"]]
            .dropna()
            .drop_duplicates(subset=["player_id"], keep="first")
        )
        players = players.merge(headshots, on="player_id", how="left")

    # 2) Compute team win% from played games only
    sched_pl = nfl.load_schedules([season])
    sched = sched_pl.to_pandas()

    played = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()

    home = pd.DataFrame({
        "season": played["season"],
        "team": played["home_team"],
        "win":  (played["home_score"] > played["away_score"]).astype(int),
    })
    away = pd.DataFrame({
        "season": played["season"],
        "team": played["away_team"],
        "win":  (played["away_score"] > played["home_score"]).astype(int),
    })
    team_games = pd.concat([home, away], ignore_index=True)

    standings = (
        team_games.groupby(["season", "team"], as_index=False)
        .agg(games=("win", "size"), wins=("win", "sum"))
    )
    standings["win_pct"] = standings["wins"] / standings["games"]

    # Merge win_pct to players on (season, team)
    if "team" not in players.columns:
        # Fallback: try 'recent_team'
        if "recent_team" in players.columns:
            players = players.rename(columns={"recent_team": "team"})
        else:
            raise KeyError(f"No team column found in players. Available: {players.columns.tolist()}")

    players = players.merge(
        standings[["season", "team", "win_pct"]],
        on=["season", "team"],
        how="left",
    )
    players["win_pct"] = pd.to_numeric(players["win_pct"], errors="coerce").fillna(0.0).clip(0, 1)

    # 3) Keys, engineered features
    name_src = "player_display_name" if "player_display_name" in players.columns else "player_name"
    players["player_name_key"] = _clean_name(players[name_src])

    # Keep headshots if available
    headshot_cols = [c for c in ["headshot_url", "headshot"] if c in players.columns]

    df = _add_derived_features(players)

    # 4) Build X EXACTLY like training, then predict with Logistic Regression
    if not TRAINING_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAINING_METRICS_PATH}. Train first: python -m src.modeling.train_model"
        )
    with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
        tm = json.load(f)

    features_used = tm.get("features_used")
    if not isinstance(features_used, list) or not features_used:
        raise ValueError(f"'features_used' missing/invalid in {TRAINING_METRICS_PATH}")

    X = df.reindex(columns=features_used, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train with: python -m src.modeling.train_model"
        )
    model = load(MODEL_PATH)

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        dbg = {
            "model_type": type(model).__name__,
            "X_shape": X.shape,
            "first_5_features": X.columns[:5].tolist(),
            "nans_in_X": int(np.isnan(X.to_numpy()).sum()),
        }
        raise RuntimeError(f"predict_proba failed: {e}; debug={dbg}") from e

    df["mvp_probability"] = probs
    assert "mvp_probability" in df.columns, "Internal error: mvp_probability not set"

    # 5) Realism filters & presentation
    view = df.copy()
    view = view[view["games"] >= 8]                      # minimum sample size
    view = view[view["position"].isin(["QB", "RB"])]     # soft stance: QBs + elite RBs

    # Optional soft position weighting
    pos_weight = {"QB": 1.0, "RB": 0.85}
    view["adjusted_pred"] = view["mvp_probability"] * view["position"].map(pos_weight).fillna(0.6)

    # Local team logo path for Canva CSV merge (drop PNGs in assets/logos as TEAM_ABBR.png)
    view["team_logo_url"] = CDN_BASE + "/assets/logos/" + view["team"].astype(str) + ".png"

    # --- Optional rollups for a richer table ---
    view["total_yards"] = (
        view.get("passing_yards", pd.Series(0)).astype(float).fillna(0)
        + view.get("rushing_yards", pd.Series(0)).astype(float).fillna(0)
        + view.get("receiving_yards", pd.Series(0)).astype(float).fillna(0)
    )

    view["total_tds"] = (
        view.get("passing_tds", pd.Series(0)).astype(float).fillna(0)
        + view.get("rushing_tds", pd.Series(0)).astype(float).fillna(0)
        + view.get("receiving_tds", pd.Series(0)).astype(float).fillna(0)
    )

    # Per-game rates were already computed: pass_ypg, rush_ypg, recv_ypg


        # --- Choose what to show in the CSVs / Canva merge ---
    display_cols = [
        # identity / context
        "season", name_src, "position", "team",
        "games", "win_pct",

        # efficiency & engineered
        "td_int_ratio", "win_pct_scaled", "games_weighted_win", "efficiency_win",

        # volume (season totals)
        "passing_yards", "passing_tds", "passing_interceptions",
        "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds",

        # rollups
        "total_yards", "total_tds",

        # per-game
        "pass_ypg", "rush_ypg", "recv_ypg",

        # model outputs
        "mvp_probability", "adjusted_pred","headshot_url","team_logo_url"
    ] + headshot_cols + ["team_logo_path"]


    display = (
        view.sort_values("adjusted_pred", ascending=False)
            .loc[:, [c for c in display_cols if c in view.columns]]
    )

    # Sort + rank within view
    display = display.sort_values("adjusted_pred", ascending=False).reset_index(drop=True)
    display.insert(0, "rank", display.index + 1)

    # Pretty display columns
    display["win_pct_disp"] = (display["win_pct"] * 100).round(0).astype(int).astype(str) + "%"
    display["td_int_disp"]  = display["td_int_ratio"].round(2)
    display["pass_ypg"]     = pd.Series(display.get("pass_ypg", 0)).round(1)
    display["total_yards"]  = pd.Series(display.get("total_yards", 0)).round(0).astype(int)
    display["prob_disp"]    = (display["adjusted_pred"] * 100).clip(0, 100).round(0).astype(int).astype(str) + "%"

    # Normalize name and logo path columns
    name_src = "player_display_name" if "player_display_name" in display.columns else "player_name"
    display = display.rename(columns={name_src: "player_name"})

    # Keep only the columns Canva needs (but don’t break your full CSVs)
    canva_cols = [
        "rank","player_name","team","position","games",
        "win_pct_disp","td_int_disp","pass_ypg","total_yards","prob_disp",
        "headshot_url","team_logo_url"
    ]
    canva = display[[c for c in canva_cols if c in display.columns]].copy()

    # Save a specific Canva CSV too
    OUT_CANVA = OUTPUTS_DIR / f"mvp_watch_canva_{SEASON}.csv"
    canva.to_csv(OUT_CANVA, index=False)

    # 6) Save outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    display.to_csv(OUT_ALL, index=False)
    display.head(10).to_csv(OUT_TOP10, index=False)
    display.head(5).to_csv(OUT_TOP5, index=False)

    print(f"Saved ALL   → {OUT_ALL}")
    print(f"Saved Top10 → {OUT_TOP10}")
    print(f"Saved Top5  → {OUT_TOP5}")
    print(display.head(10).to_string(index=False))

    return str(OUT_TOP10)


if __name__ == "__main__":
    run()

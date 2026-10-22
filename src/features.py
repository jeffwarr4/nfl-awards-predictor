"""
Shared feature engineering for training (src/modeling/train_model.py) and
in-season scoring (src/Predictions/predict_current.py). Both must build the
model's features identically, so the logic lives only here.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def clean_name(s_like) -> pd.Series:
    """
    Normalize player names for joining:
    - lowercase
    - remove middle initials (single-letter tokens)
    - drop common suffixes (jr, sr, ii, iii, iv, v)
    - remove punctuation
    - collapse whitespace
    """
    def norm(x: str) -> str:
        x = str(x).strip().lower()
        x = _PUNCT_RE.sub(" ", x)
        # remove single-letter middle initials (e.g., "lamar d jackson" -> "lamar jackson")
        tokens = [t for t in x.split() if len(t) > 1 or not t.isalpha()]
        x = " ".join(tokens)
        x = _SUFFIX_RE.sub("", x)
        return re.sub(r"\s+", " ", x).strip()

    return pd.Series(s_like).astype(str).map(norm)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game rates, TD:INT, position flags and the win% interactions."""
    df = df.copy()

    if "games" not in df.columns:
        df["games"] = 0
    df["games"] = pd.to_numeric(df["games"], errors="coerce").fillna(0).astype(float)

    if "win_pct" not in df.columns:
        df["win_pct"] = 0.0
    df["win_pct"] = pd.to_numeric(df["win_pct"], errors="coerce").fillna(0.0).clip(0, 1)

    # Per-game yards (safe divide)
    games_nonzero = df["games"].replace(0, np.nan)
    for src, tgt in [("passing_yards", "pass_ypg"),
                     ("rushing_yards", "rush_ypg"),
                     ("receiving_yards", "recv_ypg")]:
        if src in df.columns:
            df[tgt] = (pd.to_numeric(df[src], errors="coerce") / games_nonzero).fillna(0.0)
        else:
            df[tgt] = 0.0

    # TD:INT ratio (robust to INT=0)
    if {"passing_tds", "passing_interceptions"}.issubset(df.columns):
        tds = pd.to_numeric(df["passing_tds"], errors="coerce")
        denom = pd.to_numeric(df["passing_interceptions"], errors="coerce").replace(0, np.nan)
        df["td_int_ratio"] = (tds / denom).fillna(tds).astype(float)
    else:
        df["td_int_ratio"] = 0.0

    # Position flags
    if "position" in df.columns:
        df["is_qb"] = (df["position"] == "QB").astype(int)
        df["is_rb"] = (df["position"] == "RB").astype(int)
        df["is_wr"] = (df["position"] == "WR").astype(int)
    else:
        df["is_qb"] = df["is_rb"] = df["is_wr"] = 0

    # Team-success interactions
    df["win_pct_scaled"]     = df["win_pct"] * 10.0
    df["games_weighted_win"] = df["games"] * df["win_pct"]
    df["efficiency_win"]     = df["td_int_ratio"] * df["win_pct_scaled"]

    return df


def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Exactly the model's columns, numeric, NaN/inf → 0."""
    X = df.reindex(columns=features, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

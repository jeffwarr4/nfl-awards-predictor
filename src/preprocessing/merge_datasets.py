"""
Merge player season stats (features) with AP MVP winners (targets)
to create a first-pass training dataset.

Input:
  data/raw/player_season_stats.csv
  data/raw/mvp_winners.csv

Output:
  data/processed/training_data.csv

Run:
  python -m src.preprocessing.merge_datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd


def _clean_name(s: pd.Series) -> pd.Series:
    """
    Normalize player names for joining:
    - lowercase
    - remove middle initials (single-letter tokens)
    - drop common suffixes (jr, sr, ii, iii, iv, v)
    - remove punctuation
    - collapse whitespace
    """
    import re

    suffix_re = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", flags=re.IGNORECASE)
    # remove punctuation except spaces
    punct_re = re.compile(r"[^\w\s]")

    def norm(x: str) -> str:
        x = str(x).strip().lower()
        x = punct_re.sub(" ", x)
        # remove single-letter middle initials (e.g., "lamar d jackson" -> "lamar jackson")
        tokens = [t for t in x.split() if len(t) > 1 or not t.isalpha()]
        x = " ".join(tokens)
        # drop suffixes
        x = suffix_re.sub("", x)
        # collapse multi-space
        x = re.sub(r"\s+", " ", x).strip()
        return x

    return s.astype(str).map(norm)



def run(
    player_stats_path: Optional[Union[str, Path]] = None,
    winners_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> str:
    repo_root = Path(__file__).resolve().parents[2]

    player_stats_path = Path(player_stats_path) if player_stats_path else repo_root / "data" / "raw" / "player_season_stats.csv"
    winners_path = Path(winners_path) if winners_path else repo_root / "data" / "raw" / "mvp_winners.csv"
    out_path = Path(out_path) if out_path else repo_root / "data" / "processed" / "training_data.csv"

    # --- Load
    players = pd.read_csv(player_stats_path)
    winners = pd.read_csv(winners_path)

    # --- Basic sanity
    required_player_cols = {"season", "player_name"}
    missing_player = required_player_cols - set(players.columns)
    if missing_player:
        raise ValueError(f"Players file missing columns: {missing_player}")

    required_winner_cols = {"season", "player_name", "mvp_winner"}
    missing_winner = required_winner_cols - set(winners.columns)
    if missing_winner:
        raise ValueError(f"Winners file missing columns: {missing_winner}")

    # --- Normalize join keys (prefer full display name)
    player_name_source = "player_display_name" if "player_display_name" in players.columns else "player_name"
    players["player_name_for_key"] = players[player_name_source]
    players["player_name_key"] = _clean_name(players["player_name_for_key"])
    winners["player_name_key"] = _clean_name(winners["player_name"])

    # Trim to overlapping season range (helps avoid accidental joins)
    min_season = max(int(players["season"].min()), int(winners["season"].min()))
    max_season = min(int(players["season"].max()), int(winners["season"].max()))
    players = players[(players["season"] >= min_season) & (players["season"] <= max_season)]

    # --- Merge target flag (left join so we keep all players)
    merged = players.merge(
        winners[["season", "player_name_key", "mvp_winner"]],
        on=["season", "player_name_key"],
        how="left",
    )
    merged["mvp_winner"] = merged["mvp_winner"].fillna(0).astype(int)

    # --- Helpful derived features (safe if columns exist)
    # Per-game already created earlier; add win% buckets/position group for modeling
    if "win_pct" in merged.columns:
        merged["team_win_bin"] = pd.cut(
            merged["win_pct"],
            bins=[-1, 0.25, 0.5, 0.75, 1.01],
            labels=["<=25%", "25-50%", "50-75%", "75-100%"],
        )

    if "position" in merged.columns:
        merged["is_qb"] = (merged["position"] == "QB").astype(int)
        merged["is_rb"] = (merged["position"] == "RB").astype(int)
        merged["is_wr"] = (merged["position"] == "WR").astype(int)

    # --- Output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    winners_found = int(merged["mvp_winner"].sum())
    seasons = int(merged["season"].nunique())
    print(f"Wrote {out_path} | Seasons={seasons} | Winners matched={winners_found}")

    return str(out_path)


if __name__ == "__main__":
    run()
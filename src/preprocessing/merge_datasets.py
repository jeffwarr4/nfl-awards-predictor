"""
Merge player season stats (features) with AP MVP winners and vote tables
(targets) to create the training dataset.

Input:
  data/raw/player_season_stats.csv
  data/raw/mvp_winners.csv
  data/raw/mvp_votes.csv     (season, vote_rank, player_name, position, votes, share)

Targets written:
  mvp_winner  1 if the player won MVP that season
  any_votes   1 if the player received any MVP vote that season
  vote_share  the player's AP vote share (0 for non-vote-getters)

Output:
  data/processed/training_data.csv

Run:
  python -m src.preprocessing.merge_datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.features import clean_name
from config import PLAYER_STATS_CSV, MVP_WINNERS_CSV, MVP_VOTES_CSV, TRAINING_CSV


def run(
    player_stats_path: Optional[Union[str, Path]] = None,
    winners_path: Optional[Union[str, Path]] = None,
    votes_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> str:
    player_stats_path = Path(player_stats_path) if player_stats_path else PLAYER_STATS_CSV
    winners_path = Path(winners_path) if winners_path else MVP_WINNERS_CSV
    votes_path = Path(votes_path) if votes_path else MVP_VOTES_CSV
    out_path = Path(out_path) if out_path else TRAINING_CSV

    # --- Load
    players = pd.read_csv(player_stats_path)
    winners = pd.read_csv(winners_path)
    votes = pd.read_csv(votes_path)

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
    players["player_name_key"] = clean_name(players["player_name_for_key"])
    winners["player_name_key"] = clean_name(winners["player_name"])
    votes["player_name_key"] = clean_name(votes["player_name"])

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

    # --- Vote labels (any vote + vote share)
    merged = merged.merge(
        votes[["season", "player_name_key", "share"]].rename(columns={"share": "vote_share"}),
        on=["season", "player_name_key"],
        how="left",
    )
    merged["any_votes"] = merged["vote_share"].notna().astype(int)
    merged["vote_share"] = merged["vote_share"].fillna(0.0)

    # Same-name collisions (e.g. two Adrian Petersons in 2008): only the
    # higher-volume player actually got the votes.
    scrimmage = merged[["passing_yards", "rushing_yards", "receiving_yards"]].fillna(0).sum(axis=1)
    top_row = scrimmage.groupby([merged["season"], merged["player_name_key"]]).transform("max") == scrimmage
    dupe = merged.duplicated(["season", "player_name_key"], keep=False)
    for col in ("mvp_winner", "any_votes", "vote_share"):
        merged.loc[dupe & ~top_row, col] = 0

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

    in_range = votes[votes["season"].between(min_season, max_season)]
    matched = merged.loc[merged["any_votes"] == 1, ["season", "player_name_key"]]
    unmatched = in_range.merge(matched, on=["season", "player_name_key"], how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    print(f"Vote-getters matched={len(matched)} of {len(in_range)}")
    for _, r in unmatched.iterrows():
        print(f"  unmatched vote-getter: {r['season']} {r['player_name']}")

    return str(out_path)


if __name__ == "__main__":
    run()
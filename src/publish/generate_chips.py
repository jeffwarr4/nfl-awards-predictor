"""
Generate the "Chip" (3 pipe-separated tags) and "Why" (one-line rationale) for
the top-3 players on the NFL MVP board, using the Claude API.

Port of mlb-awards-predictor/src/generate_chips.py.

Output:
  predictions/{year}/chip_data/chip_data_latest.csv   (player_key, Chip, Why)
  predictions/{year}/chip_data/chip_data_{ts}.csv     (+ rank, for reference)

Skipped with a warning when ANTHROPIC_API_KEY is not set.

Run standalone (from the latest board):
    python -m src.publish.generate_chips
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import CURRENT_PRED_DIR
from src.publish.stats import stat_line

# Haiku: fast, cheap, no thinking blocks, reliable at strict JSON for 3-word tags.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

CHIP_SYSTEM_PROMPT = """You generate two short labels for NFL awards-prediction graphics: a "Chip" and a "Why".

Rules:
- "chip": exactly 3 short tags (1-3 words each), separated by " | ". Each tag names a distinct statistical strength (e.g. "Efficient passer", "Ball security", "Dual threat", "Workhorse back", "Winning team").
- "why": one sentence, under 100 characters, giving the stat-driven reason for the player's rank. No fluff, no adjectives that aren't backed by a stat in the input.
- Base every word ONLY on the stats given in the user message. Never invent, infer, or assume a stat that wasn't provided.
- Tags describe skill/performance, not identity — don't put the player's name, team, or rank number in the chip itself.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{"chip": "Tag1 | Tag2 | Tag3", "why": "One sentence explanation."}"""


def build_user_message(player: pd.Series) -> str:
    return (f"Player: {player['player_name']} ({player['team']}, {player['position']})\n"
            f"Rank: #{int(player['rank'])} in our model's NFL MVP prediction\n"
            f"Games: {int(player['games'])} of {int(player['team_games'])} team games\n"
            f"Stats: {stat_line(player)}")


def generate_chip_for_player(client, player: pd.Series, retries: int = 2) -> dict:
    message = build_user_message(player)
    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=CHIP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
            )
            block = next((b for b in resp.content if hasattr(b, "text")), None)
            if block is None:
                raise ValueError("No text block in response")
            data = json.loads(re.sub(r"^```json|```$", "", block.text.strip()).strip())
            return {"chip": data["chip"], "why": data["why"]}
        except Exception as e:
            if attempt == retries:
                print(f"  ! Chip generation failed for {player.get('player_name', '?')}: {e}")
                return {"chip": "", "why": ""}
            time.sleep(1.5)


def generate_chips(top10: pd.DataFrame, outdir: Path, top_n: int = 3) -> pd.DataFrame:
    """Chips for the top_n players. Returns (player_key, Chip, Why); empty if no API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — chip generation skipped")
        return pd.DataFrame(columns=["player_key", "Chip", "Why"])

    from anthropic import Anthropic
    client = Anthropic()

    rows = []
    for _, player in top10.sort_values("rank").head(top_n).iterrows():
        print(f"  Generating chip: {player['player_name']}")
        result = generate_chip_for_player(client, player)
        rows.append({"player_key": player["player_key"], "Chip": result["chip"],
                     "Why": result["why"], "rank": int(player["rank"])})
    chip_df = pd.DataFrame(rows)

    chip_dir = Path(outdir) / "chip_data"
    chip_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chip_df[["player_key", "Chip", "Why"]].to_csv(chip_dir / "chip_data_latest.csv", index=False)
    chip_df.to_csv(chip_dir / f"chip_data_{ts}.csv", index=False)
    print(f"Chip data -> {chip_dir / 'chip_data_latest.csv'}")
    return chip_df[["player_key", "Chip", "Why"]]


if __name__ == "__main__":
    generate_chips(pd.read_csv(CURRENT_PRED_DIR / "top10_mvp_latest.csv"), CURRENT_PRED_DIR)

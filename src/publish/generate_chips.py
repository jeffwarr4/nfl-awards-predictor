"""
Generate the "Chip" (3 pipe-separated tags) and "Why" (one-line rationale) for
the top-3 players on the NFL MVP board, plus a short Why (no chip) for the
"On the Bubble" ranks (config.BUBBLE_RANKS), using the Claude API.

Port of mlb-awards-predictor/src/generate_chips.py.

Output:
  predictions/{year}/chip_data/chip_data_latest.csv   (player_key, Chip, Why; Chip blank for bubble rows)
  predictions/{year}/chip_data/chip_data_{ts}.csv     (+ rank, for reference)

Skipped with a warning when ANTHROPIC_API_KEY is not set.

Run standalone (from the latest 30-deep board, else the top 10):
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

from config import BUBBLE_RANKS, CURRENT_PRED_DIR, TOP_N
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


BUBBLE_WHY_MAX = 45   # fits the smallest bubble label plate (#15, 300 px)

BUBBLE_SYSTEM_PROMPT = f"""You write a one-line "Why" for players ranked just outside the top 10 of an NFL MVP prediction board ("On the Bubble").

Rules:
- At most {BUBBLE_WHY_MAX} characters, including spaces. Shorter is better. No ending period.
- Stats only: use ONLY the numbers given in the user message. Never invent, infer, or assume a stat.
- When the stats show it, say what is keeping the player out of the top 10 (e.g. team record, TD total, vs the #10 player's line). Otherwise name their best stat.
- No player name, no rank number, no hype adjectives.
- Output ONLY valid JSON in exactly this shape, nothing else, no markdown fences, no preamble:
{{"why": "Short line"}}"""


def _record(player: pd.Series) -> str:
    """W-L from win_pct × team_games (ties count as non-wins, as in team_records)."""
    wins = int(round(float(player["win_pct"]) * int(player["team_games"])))
    return f"{wins}-{int(player['team_games']) - wins}"


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


def build_bubble_message(player: pd.Series, cutoff: pd.Series | None) -> str:
    msg = (f"Player: {player['player_name']} ({player['team']}, {player['position']})\n"
           f"Rank: #{int(player['rank'])}, just outside the top 10\n"
           f"Team record: {_record(player)}\n"
           f"Games: {int(player['games'])} of {int(player['team_games'])} team games\n"
           f"Stats: {stat_line(player)}")
    if cutoff is not None:
        msg += (f"\n\nFor comparison, #{int(cutoff['rank'])} {cutoff['player_name']} "
                f"({cutoff['position']}, team record {_record(cutoff)}): {stat_line(cutoff)}")
    return msg


def _ask_why(client, message: str) -> str:
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=100,
        system=BUBBLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )
    block = next((b for b in resp.content if hasattr(b, "text")), None)
    if block is None:
        raise ValueError("No text block in response")
    return str(json.loads(re.sub(r"^```json|```$", "", block.text.strip()).strip())["why"]).strip()


def generate_bubble_why(client, player: pd.Series, cutoff: pd.Series | None) -> str:
    """≤ BUBBLE_WHY_MAX chars: one retry with the length called out, then hard-truncate."""
    message = build_bubble_message(player, cutoff)
    try:
        why = _ask_why(client, message)
        if len(why) > BUBBLE_WHY_MAX:
            why = _ask_why(client, f"{message}\n\nYour last answer was {len(why)} characters: "
                                   f'"{why}". Rewrite it in {BUBBLE_WHY_MAX} characters or fewer.')
    except Exception as e:
        print(f"  ! Bubble Why failed for {player.get('player_name', '?')}: {e}")
        return ""
    if len(why) > BUBBLE_WHY_MAX:
        why = why[:BUBBLE_WHY_MAX - 1].rstrip(" ,;·-") + "…"
    return why


def generate_chips(board: pd.DataFrame, outdir: Path, top_n: int = 3) -> pd.DataFrame:
    """Chips + Why for the top_n players, Why only for ranks BUBBLE_RANKS.
    Returns (player_key, Chip, Why); empty if no API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY not set — chip generation skipped")
        return pd.DataFrame(columns=["player_key", "Chip", "Why"])

    from anthropic import Anthropic
    client = Anthropic()

    board = board.sort_values("rank")
    rows = []
    for _, player in board.head(top_n).iterrows():
        print(f"  Generating chip: {player['player_name']}")
        result = generate_chip_for_player(client, player)
        rows.append({"player_key": player["player_key"], "Chip": result["chip"],
                     "Why": result["why"], "rank": int(player["rank"])})

    cutoff = board[board["rank"] == TOP_N]
    cutoff = cutoff.iloc[0] if not cutoff.empty else None
    for _, player in board[board["rank"].between(*BUBBLE_RANKS)].iterrows():
        print(f"  Generating bubble Why: {player['player_name']}")
        rows.append({"player_key": player["player_key"], "Chip": "",
                     "Why": generate_bubble_why(client, player, cutoff), "rank": int(player["rank"])})
    chip_df = pd.DataFrame(rows)

    chip_dir = Path(outdir) / "chip_data"
    chip_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chip_df[["player_key", "Chip", "Why"]].to_csv(chip_dir / "chip_data_latest.csv", index=False)
    chip_df.to_csv(chip_dir / f"chip_data_{ts}.csv", index=False)
    print(f"Chip data -> {chip_dir / 'chip_data_latest.csv'}")
    return chip_df[["player_key", "Chip", "Why"]]


if __name__ == "__main__":
    deep = CURRENT_PRED_DIR / "board_mvp_latest.csv"
    generate_chips(pd.read_csv(deep if deep.exists() else CURRENT_PRED_DIR / "top10_mvp_latest.csv"),
                   CURRENT_PRED_DIR)

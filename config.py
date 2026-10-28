# config.py
# ---------------------------------------------------------------
# Single source of truth for the nfl-awards-predictor project.
# Every script imports from here — no hardcoded years or paths elsewhere.
# ---------------------------------------------------------------

import os
from pathlib import Path

# ── Active year ───────────────────────────────────────────────────
CURRENT_YEAR = 2026      # season being predicted RIGHT NOW
TRAIN_START  = 2000      # first season of training data
TRAIN_END    = 2025      # last season with a complete AP MVP vote

# ── Directory layout ──────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_RAW    = ROOT_DIR / "data" / "raw"
DATA_PROC   = ROOT_DIR / "data" / "processed"
MODELS_DIR  = ROOT_DIR / "models"
PREDICTIONS = ROOT_DIR / "predictions"
FONTS_DIR   = ROOT_DIR / "assets" / "fonts"

PLAYER_STATS_CSV = DATA_RAW / "player_season_stats.csv"
MVP_WINNERS_CSV  = DATA_RAW / "mvp_winners.csv"
MVP_VOTES_CSV    = DATA_RAW / "mvp_votes.csv"      # PFR vote tables, offensive players only
TRAINING_CSV     = DATA_PROC / "training_data.csv"

CURRENT_PRED_DIR = PREDICTIONS / str(CURRENT_YEAR)
CANDIDATES_CSV   = PREDICTIONS / "nfl_candidates.csv"   # fetched by stinger-assets

# Background for the weekly board graphic (1080×1350). If missing,
# render_graphics.py draws a plain dark placeholder.
BOARD_BACKGROUND = ROOT_DIR / "MVP Awards NFL.png"

# "On the Bubble" slide (ranks BUBBLE_RANKS). Background + overlay + slot geometry
# come from src/publish/make_bubble_bg.py; the overlay is composited over headshots.
BUBBLE_BACKGROUND = ROOT_DIR / "MVP Bubble NFL.png"
BUBBLE_OVERLAY    = ROOT_DIR / "MVP Bubble NFL overlay.png"
BUBBLE_SLOTS      = ROOT_DIR / "assets" / "bubble_slots.json"

# Sibling repo with headshots/logos. CI checks it out and sets the env var.
STINGER_ASSETS_DIR = Path(os.environ.get(
    "STINGER_ASSETS_DIR",
    str(ROOT_DIR.parent / "stinger-assets"),
))

# ── Model ─────────────────────────────────────────────────────────
# "vote_share": label = received any MVP vote, sample weight 1 + 20*share
#               for vote-getters (backtest: winner #1 58%, top-3 85%, top-5 96%).
# "winner":     label = won MVP (the 2025 model; 54% / 81% / 85%).
# Both can be trained side by side; MODEL_TARGET picks the one predict uses.
MODEL_TARGET  = "vote_share"
TARGETS       = ("vote_share", "winner")
SHARE_WEIGHT  = 20.0
RANDOM_STATE  = 42

FEATURES = [
    "games", "attempts", "completions",
    "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
    "rushing_yards", "rushing_tds", "rushing_epa",
    "receiving_yards", "receiving_tds", "receiving_epa",
    "pass_ypg", "rush_ypg", "recv_ypg",
    "win_pct",
    "td_int_ratio",
    "is_qb", "is_rb", "is_wr",
    "win_pct_scaled", "games_weighted_win", "efficiency_win",
]

def model_path(target: str = MODEL_TARGET) -> Path:
    return MODELS_DIR / f"nfl_mvp_{target}.joblib"

def metrics_path(target: str = MODEL_TARGET) -> Path:
    return MODELS_DIR / f"metrics_{target}.json"

# ── In-season scoring ─────────────────────────────────────────────
SEASON_GAMES        = 17      # counting stats are projected to this pace
MIN_TEAM_GAME_SHARE = 0.60    # must have played ≥60% of his team's games
BOARD_POSITIONS     = ("QB", "RB")
TOP_N               = 10
# Saved board depth. 2010–2025 replay: every eventual winner was top-15 by Week 8,
# every top-5 vote finisher top-26 from Week 8 on; top 30 covers 98–100% from Week 4.
BOARD_DEPTH         = 30
BUBBLE_RANKS        = (11, 15)  # inclusive; the "On the Bubble" slide
CANDIDATE_POOL      = 30      # headshots pre-queued for everyone on the saved board
ARCHIVES_TO_KEEP    = 4

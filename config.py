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
BOARD_POSITIONS     = ("QB", "RB", "WR")   # WR added 2026: backtest winner ranks unchanged
TOP_N               = 10
# Saved board depth. 2010–2025 replay: every eventual winner was top-15 by Week 8,
# every top-5 vote finisher top-26 from Week 8 on; top 30 covers 98–100% from Week 4.
BOARD_DEPTH         = 30
BUBBLE_RANKS        = (11, 15)  # inclusive; the "On the Bubble" slide
CANDIDATE_POOL      = 30      # headshots pre-queued for everyone on the saved board
ARCHIVES_TO_KEEP    = 4

# ── Weekly reel (src/publish/render_reel.py) ──────────────────────
# One story per week as a silent 1080×1920 MP4 plus a narration script Jeff voices over in Instagram.
# Reworked Nov 2026: the first reels lost most viewers in the hook (NHL: ~6 of 27 s watched), so the video is
# ~20 s, frame 0 already shows the player, rank and hook, and it ends on a comment question, not a spoken URL.
REEL_MODEL     = "claude-opus-5-5"   # narration script (chips/captions stay on Haiku)
REEL_SIZE      = (1080, 1920)
REEL_FPS       = 30
# (scene, start s, end s). The script's beats are labelled with these windows, so they stay in sync.
REEL_SCENES    = (("hook", 0, 3), ("proof", 3, 13), ("context", 13, 16), ("ask", 16, 20))
REEL_HOLD_S    = 1         # last frame held so a slightly long voiceover isn't cut off
REEL_SAFE_TOP    = 250     # px kept clear of the platform UI (top bar)
REEL_SAFE_BOTTOM = 400     # px kept clear at the bottom (caption, buttons)
REEL_SAFE_LEFT   = 60
REEL_SAFE_RIGHT  = 120     # Instagram's like/comment/share column
REEL_WORDS     = (40, 50)  # narration length across all four beats
REEL_PROOF_MAX_WORDS = 26  # the proof beat: short sentences, one fact each (a 31-word NBA proof read as wordy)
REEL_WPS       = 3.0       # speaking pace for each beat's word target
REEL_HOOK_MAX_WORDS = 10   # the first beat: his name, then the hook fact ("Dak Prescott: No. 17 to No. 10 in three weeks.")
REEL_ASK_MAX_WORDS  = 14   # the last beat: one comment question
# Story picker thresholds (src/publish/reel_story.py)
REEL_TOP_SPOTS = 5         # "top 5" for the climber and drop stories
REEL_MIN_CLIMB = 2         # spots gained to count as a top-5 climber
REEL_MIN_DROP  = 3         # spots lost by a former top-5 player to count as a drop
REEL_LOOKBACK_WEEKS = 3    # the "then" in then-vs-now: up to this many weeks back (or the path hook's week)
# What drove the move, told to the script (reel_script.story_frame): a key rate for his position (stats.RATE_COLS)
# that moved at least its REEL_FLAT_STATS amount (absolute) with his rank makes it his own production; otherwise a
# team win rate in the window at least REEL_TEAM_SWING above (or below) its rate before the window, over 3+ team
# games, makes it the team. 2026 Week 11: Dak's rating 93.2 -> 101.0 (+7.8) is his own production.
REEL_FLAT_STATS = {"passer_rating": 5.0, "td_int": 1.0, "pass_ypg": 20.0, "total_ypg": 15.0, "rush_ypg": 10.0}
REEL_TEAM_SWING = 0.25
# Players fans know by a short name: saying it counts as saying the full name (accents don't matter).
REEL_NICKNAMES: dict = {}

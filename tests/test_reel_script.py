"""Reel script checks (no API calls). Run from the repo root: python -m pytest tests/"""

import copy

import pandas as pd

from src.publish.reel_script import (
    caption_with_cta, context_problems, driver, fallback_question, finish, milestone, probability_language, problems, opener_problems,
    proof_opener, story_frame, unanchored_time,
)
from src.publish.reel_story import Story

NAMES = ["Brock Purdy", "Joe Burrow", "Jahmyr Gibbs", "Josh Allen", "Patrick Mahomes II", "Lamar Jackson",
         "Jordan Love", "Jalen Hurts", "Baker Mayfield", "Dak Prescott", "Jared Goff", "C.J. Stroud"]
TEAMS = ["SF", "CIN", "DET", "BUF", "KC", "BAL", "GB", "PHI", "TB", "DAL", "DET", "HOU"]
BOARD = pd.DataFrame({"rank": range(1, 13), "player_name": NAMES, "team": TEAMS, "player_id": [f"id{i}" for i in range(1, 13)]})


def row(name):
    return BOARD[BOARD["player_name"] == name].iloc[0]


def story(kind="bubble_breakthrough", name="Dak Prescott", prev=17):
    r = row(name)
    return Story(kind, r, prev, int(r["rank"]), "Into the top 10", "")


# Example 2 from the prompt (a production-driven climb), as edited to pass the checks
GOOD = {
    "beats": [
        {"scene": "hook", "text": "Dak Prescott jumped from 17 to 10 in three weeks."},
        {"scene": "proof", "text": "So what changed? His last three games: twelve touchdowns, two Dallas wins, and "
                                   "his season rating up from 93 to 101."},
        {"scene": "context", "text": "Five spots from our top five, and Mahomes is the one to catch."},
        {"scene": "ask", "text": "Can Dak crack the top 5?"},
    ],
    "question": "Can Dak crack the top 5?",
    "instagram_caption": "Can Dak crack the top 5? Full board at stingercollectibles.com",
    "hashtags": ["#StingerNFL", "#NFLMVP", "#DakPrescott", "#DallasCowboys", "#NFLWeek11"],
}

# What went out for 2026 Week 11 (v2)
WEEK11_V2 = {
    "beats": [
        {"scene": "hook", "text": "No. 17 to No. 10 in one week."},
        {"scene": "proof", "text": "His passing got sharper. Rating about 93 then. About 101 now. Twelve touchdowns in "
                                   "three games. Dallas went 2-1 since. Our model rewards that."},
        {"scene": "context", "text": "He's No. 10 now. Brock Purdy holds No. 1."},
        {"scene": "ask", "text": "Can Dak climb into the top 5?"},
    ],
    "question": "Can Dak climb into the top 5?",
    "instagram_caption": GOOD["instagram_caption"], "hashtags": GOOD["hashtags"],
}


def script(**changes):
    d = copy.deepcopy(GOOD)
    for scene, text in changes.items():
        next(b for b in d["beats"] if b["scene"] == scene)["text"] = text
    return d


def check(d, s=None, ranks=None):
    return problems(d, s or story(), BOARD, {"ranks": ranks or {17, 10}})


def test_example_2_passes():
    assert check(GOOD) == []


def test_example_1_passes():
    ex1 = {"beats": [
        {"scene": "hook", "text": "Patrick Mahomes jumped from 30 to 5 in three weeks."},
        {"scene": "proof", "text": "His numbers barely moved: twenty-eight points and nearly thirteen boards a night. "
                                   "So what changed? Kansas City, 1-4 three weeks ago, has gone 8-2 since."},
        {"scene": "context", "text": "Two spots from our top three, Gibbs next."},
        {"scene": "ask", "text": "Mahomes or Gibbs: who finishes higher?"}],
        "question": "Mahomes or Gibbs: who finishes higher?", "instagram_caption": GOOD["instagram_caption"],
        "hashtags": GOOD["hashtags"]}
    assert check(ex1, story("top5_climber", "Patrick Mahomes II", prev=30), {30, 5}) == []


def test_week11_v2_fails_the_new_checks():
    p = " || ".join(check(WEEK11_V2))
    assert "full name" in p and "hook must name him" in p
    assert "Rating about 93 then." in p and "About 101 now." in p and "Dallas went 2-1 since." in p
    assert "repeats the hook's rank (No. 10)" in p and "board leader (Brock Purdy)" in p


def test_suffix_is_optional_in_the_full_name():
    s = story("top5_climber", "Patrick Mahomes II", prev=9)
    d = script(hook="Patrick Mahomes jumped from 9 to 5 in three weeks.", context="Two spots from our top three.",
               ask="Can Mahomes crack the top 3?")
    d["question"] = "Can Mahomes crack the top 3?"
    assert check(d, s, {9, 5}) == []


def test_time_anchor_heuristic():
    assert unanchored_time("His season rating is up from 93 to 101 since Week 8.") == []
    assert unanchored_time("Dallas has gone 2-1 in those three games.") == []
    assert unanchored_time("Rating about 93 then. About 101 now.") == ["Rating about 93 then.", "About 101 now."]


def test_context_rules():
    assert context_problems("Five spots from our top five.", story(), BOARD, {17, 10}) == []
    assert any("No. 17" in x for x in context_problems("Up from No. 17, five spots from the top five.", story(), BOARD, {17, 10}))
    assert any("Brock Purdy" in x for x in context_problems("Purdy still leads.", story(), BOARD, {17, 10}))
    # a No. 1 story, or a No. 2-3 player, may name the leader
    assert context_problems("Burrow is one spot behind.", story("back_to_no1", "Brock Purdy", prev=2), BOARD, {2, 1}) == []
    assert context_problems("One spot behind Purdy.", story("top5_climber", "Joe Burrow", prev=6), BOARD, {6, 2}) == []


def test_stock_phrases_and_my_model():
    p = check(script(proof="I'll be honest, my model says his last three games were huge: twelve touchdowns and "
                           "two Dallas wins."))
    assert any("i'll be honest" in x and "my model" in x for x in p)


def test_word_range_40_to_50():
    assert any("must total" in x for x in check(script(proof="So what changed? His last three games.")))


def test_ending_must_be_a_question():
    assert any("question mark" in x for x in check(script(ask="Full board at stingercollectibles.com.")))


def test_finish_adds_milestone_question():
    d = finish(script(ask="That's the board this week."), story(), BOARD)
    assert d["question"] == "Can Prescott crack the top 5?"
    assert (d["beats"][0]["start"], d["beats"][-1]["end"]) == (0, 20)


def test_fallback_questions_by_story():
    assert fallback_question(story("new_no1", "Patrick Mahomes II")) == "Is Mahomes your MVP right now?"   # suffix dropped
    assert fallback_question(story("top5_drop")) == "Is Prescott out of the MVP race?"
    assert fallback_question(story("top5_climber", "Jahmyr Gibbs", prev=8), BOARD) == "Can Gibbs catch Purdy?"


def test_milestone_for_no10():
    m = milestone(story(), BOARD)
    assert (m["target"], m["gap"], m["other"]["player_name"]) == (5, 5, "Patrick Mahomes II")


def _row(**kw):
    base = {"player_name": "Dak Prescott", "player_id": "d", "team": "DAL", "position": "QB", "rank": 10,
            "games": 11, "team_games": 11, "win_pct": 5 / 11, "passer_rating": 101.0, "td_int": 4.17,
            "pass_ypg": 277.2, "total_tds": 27, "total_yards": 3167, "rush_yards": 118}
    base.update(kw)
    return pd.Series(base)


def _week8():
    # Dak on the 2026 Week 8 board
    return _row(week=8, rank=17, games=8, team_games=8, win_pct=0.375, passer_rating=93.2, td_int=2.8,
                pass_ypg=255.8, total_tds=15, total_yards=2147)


def test_driver_week11_is_his_production():
    d = driver(_row(), _week8(), 17)
    assert d["driver"] == "player" and d["biggest"]["col"] == "passer_rating"     # +7.8 vs a 5-point threshold
    assert d["team"]["window"] == "2-1"


def test_driver_team_and_unclear():
    flat = _row(passer_rating=94.0, td_int=3.0, pass_ypg=260.0, team_games=14, win_pct=8 / 14)  # 3-5, then 5-1
    d = driver(flat, _week8(), 17)
    assert d["driver"] == "team" and d["team"]["window"] == "5-1"
    cold = flat.copy()
    cold["win_pct"] = 4 / 14                                      # 3-5, then 1-5: the team went the other way
    assert driver(cold, _week8(), 17)["driver"] == "unclear"


def test_then_now_lines():
    from src.publish.reel_script import then_now_lines
    then = _row(week=8, rank=17, games=8, team_games=8, win_pct=2 / 8, passer_rating=95.0, td_int=2.0,
                total_tds=17, total_yards=2300)
    text = "\n".join(then_now_lines(_row(), then, 11))
    assert "Week 8, 3 weeks ago" in text and "No. 17 -> No. 10" in text
    assert "Team record: 2-6 -> 5-6 (3-0 in the window)" in text
    assert "TD to INT ratio 2.0 -> 4.2" in text and "TD:INT" not in text
    assert "In the window: 10 total touchdowns and 867 total yards in 3 games" in text


def test_then_now_skips_record_after_a_trade():
    from src.publish.reel_script import then_now_lines
    text = "\n".join(then_now_lines(_row(), _row(week=8, team="NYJ", rank=17), 11))
    assert "traded" in text and "Team record" not in text


def test_then_now_cards():
    from src.publish.render_reel import ARROW_WORD, then_now_cards
    then = _row(week=8, games=8, team_games=8, win_pct=2 / 8, passer_rating=95.0, td_int=2.0)
    cards = then_now_cards(_row(td_int=3.0, passer_rating=101.2), then)
    assert cards[0] == ("DAL record", f"2-6 {ARROW_WORD} 5-6")
    assert [c[0] for c in cards[1:]] == ["TD to INT ratio", "Passer rating"]     # biggest relative change first
    wr = _row(position="WR", passer_rating=None, td_int=None, pass_ypg=None, total_yards=900, games=9)
    cards = then_now_cards(wr, _row(position="WR", week=8, team="NYJ", total_yards=500, games=6, total_tds=4))
    assert [c[0] for c in cards] == ["Yds / game", "Total TD"]                  # traded: no record card


def test_completion_pct_allowed_win_pct_not():
    for t in ("a 68% completion rate", "68% completion percentage", "completion percentage of 68"):
        assert probability_language(t) == [], t
    for t in ("a .900 win percentage", "a 90% win rate", "a 60% chance", "he's likely the MVP", "the odds"):
        assert probability_language(t), t


def test_week7_bye_is_the_board():
    # 2026 Week 7: Allen had a bye (6 games before and after) and moved 3 -> 1 because Purdy dropped 1 -> 3
    from src.publish.reel_script import story_frame
    week6 = BOARD.assign(rank=[1, 4, 6, 3, 7, 8, 9, 10, 11, 12, 2, 5])
    week7 = BOARD.assign(rank=[3, 4, 6, 1, 7, 8, 9, 10, 11, 12, 2, 5])
    allen = _row(player_name="Josh Allen", team="BUF", rank=1, games=6, team_games=6, win_pct=5 / 6, passer_rating=102.1)
    then = allen.copy()
    then["week"], then["rank"] = 6, 3
    f = story_frame(Story("new_no1", allen, 3, 1, "New No. 1", ""), week7, then, 7, week6)
    assert f["driver"]["driver"] == "board"
    assert [c["name"] for c in f["crossed"]] == ["Jared Goff", "Brock Purdy"]     # both were ahead of him; nearest first


def test_proof_opener_rotates():
    four = [proof_opener(w) for w in range(1, 5)]
    assert len(set(four)) == 4                                   # a different lead-in each week, all four in turn
    assert all(proof_opener(w) != proof_opener(w + 1) for w in range(1, 20))


def test_proof_opener_check():
    proof = next(b["text"] for b in GOOD["beats"] if b["scene"] == "proof").replace("So what changed? ", "{}")
    assert opener_problems(proof.format("Here's why. "), "Here's why.") == []
    assert opener_problems(proof.format(""), "Here's why.") == []                 # no lead-in is fine
    p = opener_problems(proof.format("So what changed? "), "Here's why.")
    assert p and "\"So what changed?\"" in p[0]
    assert opener_problems(proof.format("The reason? "), "What happened?")
    assert opener_problems("Here's why. Here's why.", "Here's why.")


def test_story_frame_carries_the_opener():
    assert story_frame(story(), BOARD, None, 11)["opener"] == proof_opener(11)


def test_caption_site_line_is_its_own_sentence():
    cta = "Full board at stingercollectibles.com."
    assert caption_with_cta("Can Dak crack the top 5? He climbed in three weeks, and Mahomes holds No. 5 — "
                            "Full board at stingercollectibles.com") == (
        "Can Dak crack the top 5? He climbed in three weeks, and Mahomes holds No. 5. " + cta)
    assert caption_with_cta("Can Aho get back into the top 3? He slid in one week, so see where he stands: Full "
                            "board at stingercollectibles.com").endswith("so see where he stands. " + cta)
    assert caption_with_cta("Can he hold No. 1? Three straight wins.") == "Can he hold No. 1? Three straight wins. " + cta


def test_finish_rewrites_the_caption_ending():
    d = copy.deepcopy(GOOD)
    d["instagram_caption"] = d["question"] + " Up seven spots: Full board at stingercollectibles.com"
    out = finish(d, story(), BOARD, {"ranks": set()})
    assert out["instagram_caption"] == d["question"] + " Up seven spots. Full board at stingercollectibles.com."


ROBOT, NATURAL = 'Dak Prescott: No. 17 to No. 10 in three weeks.', 'Dak Prescott climbed from 17 to 10 in three weeks.'


def test_hook_reads_naturally():
    p = " || ".join(check(script(hook=ROBOT)))
    assert "the way a fan would" in p
    assert not any("the way a fan would" in x for x in check(script(hook=NATURAL)))

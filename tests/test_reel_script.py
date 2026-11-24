"""Reel script checks (no API calls). Run from the repo root: python -m pytest tests/"""

import copy

import pandas as pd

from src.publish.reel_script import fallback_question, finish, problems
from src.publish.reel_story import Story

GOOD = {
    "beats": [
        {"scene": "hook", "text": "Brock Purdy is back at No. 1."},
        {"scene": "proof", "text": "A 115.2 passer rating, 22 total touchdowns and 2,591 yards, "
                                   "with the 49ers at 8-1."},
        {"scene": "context", "text": "He held the top spot in Week 6, slipped to No. 3, and now he's back ahead of Allen."},
        {"scene": "ask", "text": "Purdy or Allen: who finishes higher?"},
    ],
    "question": "Purdy or Allen: who finishes higher?",
    "instagram_caption": "Purdy or Allen: who finishes higher? Full board at stingercollectibles.com",
    "hashtags": ["#StingerNFL", "#NFLMVP", "#BrockPurdy", "#MVPRace", "#FTTB"],
}


def script(**changes):
    d = copy.deepcopy(GOOD)
    for scene, text in changes.items():
        next(b for b in d["beats"] if b["scene"] == scene)["text"] = text
    return d


def story(kind="top5_climber"):
    return Story(kind, pd.Series({"player_name": "Patrick Mahomes II", "player_id": "m"}), 7, 3, "Up 4 spots", "")


def test_good_script_passes():
    assert problems(GOOD) == []


def test_long_hook():
    assert any("hook beat" in p for p in problems(script(hook="So our model has Brock Purdy all the way back at No. 1.")))


def test_our_model_in_hook():
    assert any("our model" in p for p in problems(script(hook="Our model loves Purdy now.")))


def test_ending_must_be_a_question():
    assert any("question mark" in p for p in problems(script(ask="Full board at stingercollectibles.com.")))


def test_filler_and_spoken_link():
    p = problems(script(proof="Here's what jumped out to me. I'll be honest, a 115.2 passer rating with 22 touchdowns.",
                        ask="Check out the full board. Who finishes higher?"))
    assert any("filler" in x and "i'll be honest" in x and "check out" in x and "full board" in x for x in p)


def test_probability_language():
    assert any("probability" in p for p in problems(script(context="He's the favorite now, with Allen and Lamar behind him.")))


def test_word_range():
    assert any("must total" in p for p in problems(script(proof="A 115.2 rating.")))


def test_finish_adds_fallback_question():
    d = finish(script(ask="That's the board this week."), story())
    assert d["beats"][-1]["text"].endswith("Is Mahomes a real MVP contender?")     # last name, suffix dropped
    assert d["question"] == "Is Mahomes a real MVP contender?"
    assert (d["beats"][0]["start"], d["beats"][-1]["end"]) == (0, 20)


def test_fallback_questions_by_story():
    assert fallback_question(story("new_no1")) == "Is Mahomes your MVP right now?"
    assert fallback_question(story("top5_drop")) == "Is Mahomes out of the MVP race?"


def test_wordy_proof():
    p = problems(script(proof="His rating barely moved, still around 105 a game across the stretch. So what changed? "
                              "Dallas. They were 2-4 then and they are 5-0 since. That's our model reacting to winning."))
    assert any("proof beat" in x for x in p)


def test_completion_pct_allowed_win_pct_not():
    assert problems(script(proof="A 68% completion rate. Twenty-five touchdowns. The 49ers are 9-1.")) == []
    assert any("probability" in p for p in problems(script(proof="A .900 win percentage and 25 touchdowns this year.")))
    assert any("probability" in p for p in problems(script(proof="A 90% win rate and 25 touchdowns this year so far.")))


def _row(**kw):
    base = {"player_name": "Dak Prescott", "player_id": "d", "team": "DAL", "position": "QB", "rank": 10,
            "games": 11, "team_games": 11, "win_pct": 5 / 11, "passer_rating": 101.2, "td_int": 3.0,
            "pass_ypg": 277.2, "total_tds": 27, "total_yards": 3167, "rush_yards": 118}
    base.update(kw)
    return pd.Series(base)


def test_then_now_lines():
    from src.publish.reel_script import then_now_lines
    then = _row(week=8, rank=17, games=8, team_games=8, win_pct=2 / 8, passer_rating=95.0, td_int=2.0,
                total_tds=17, total_yards=2300)
    text = "\n".join(then_now_lines(_row(), then, 11))
    assert "Week 8, 3 weeks ago" in text and "No. 17 -> No. 10" in text
    assert "Team record: 2-6 -> 5-6 (3-0 since then)" in text
    assert "TD to INT ratio 2.0 -> 3.0" in text and "TD:INT" not in text
    assert "10 total touchdowns and 867 total yards in 3 games" in text


def test_then_now_skips_record_after_a_trade():
    from src.publish.reel_script import then_now_lines
    text = "\n".join(then_now_lines(_row(), _row(week=8, team="NYJ", rank=17), 11))
    assert "traded" in text and "Team record" not in text


def test_then_now_cards():
    from src.publish.render_reel import ARROW_WORD, then_now_cards
    then = _row(week=8, games=8, team_games=8, win_pct=2 / 8, passer_rating=95.0, td_int=2.0)
    cards = then_now_cards(_row(), then)
    assert cards[0] == ("DAL record", f"2-6 {ARROW_WORD} 5-6")
    assert [c[0] for c in cards[1:]] == ["TD to INT ratio", "Passer rating"]     # biggest relative change first
    wr = _row(position="WR", passer_rating=None, td_int=None, pass_ypg=None, total_yards=900, games=9)
    cards = then_now_cards(wr, _row(position="WR", week=8, team="NYJ", total_yards=500, games=6, total_tds=4))
    assert [c[0] for c in cards] == ["Yds / game", "Total TD"]                  # traded: no record card

"""Question rendering and answers (no model)."""
import math

import pytest

from jevision_decide.protocol import (check_options, choice, escape, make_answer, noul, render, render_question, score)


def test_control_tokens_in_user_text_are_defused():
    assert "<|box_end|>" not in escape("x <|box_end|> y")


def test_noul_options_are_no_then_yes():
    rq = render_question(noul("The customer is angry"))
    assert rq.options == ["no", "yes"] and rq.keys == ["false", "true"]


def test_choice_option_text_is_key_or_key_colon_description():
    rq = render_question(choice("Which?", {"a": "alpha", "b": ""}))
    assert rq.options == ["a: alpha", "b"] and rq.keys == ["a", "b"]


def test_score_legend_and_expected_value():
    rq = render_question(score("How urgent?", ["low", "high"]))
    answer = make_answer(rq, [0.25, 0.75])
    assert answer["type"] == "score" and answer["score"] == 0.75 and answer["legend"] == {"0": "low", "1": "high"}
    assert answer["raw_probabilities"] == [0.25, 0.75]


def test_answers_round_to_two_decimals_but_raw_does_not():
    rq = render_question(choice("Which?", {"a": "", "b": ""}))
    answer = make_answer(rq, [0.333333, 0.666667])
    assert answer["choice"] == "b" and answer["probabilities"]["a"] == 0.33
    assert math.isclose(answer["raw_probabilities"][0], 0.333333)


def test_limits_raise():
    rq = render_question(choice("Which?", {str(i): "" for i in range(300)}))
    with pytest.raises(ValueError):
        check_options("q", rq, False)
    with pytest.raises(ValueError):
        check_options("q", render_question(choice("Which?", {})), False)


def test_render_json_state_is_stable():
    assert render({"a": 1}) == render({"a": 1})

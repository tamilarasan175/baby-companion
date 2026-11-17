from zoneinfo import ZoneInfo

import pytest

from app.services import agent

TZ = ZoneInfo("America/Edmonton")


@pytest.mark.parametrize("question, docs, visits, vaccines, logs", [
    ("My baby is not drinking milk properly she is asleep, what to do", True, False, False, False),
    ("How much did she drink today?", False, False, False, True),
    ("Enough wet diapers?", True, False, False, True),
    ("What did the doctor say yesterday?", False, True, False, False),
    ("When did she get the RSV shot?", False, False, True, False),
    ("Is it safe to give her water?", True, False, False, False),
])
def test_question_routing(question, docs, visits, vaccines, logs):
    need = agent.required_sources(question)
    assert need["docs"] == docs
    assert need["visits"] == visits
    assert need["vaccines"] == vaccines
    assert bool(need["logs"]) == logs


def test_totals_questions_need_a_totals_tool():
    need = agent.required_sources("How much did she drink today?")
    (tools, _), = need["logs"]
    assert "recent_events" not in tools


def test_safety_notice_triggers():
    for q in ["She won't wake up for feeds", "baby has a fever", "not drinking milk and very sleepy",
              "her lips look blue"]:
        assert agent.RED_FLAGS.search(q.lower()), q
    for q in ["How much did she drink today?", "When was the last dirty diaper?"]:
        assert not agent.RED_FLAGS.search(q.lower()), q


def test_checker_rejects_invented_numbers():
    evidence = '{"wet_diapers": 7, "feeds": 8}'
    assert agent.check_answer("When was the last feed?", "She had 7 wet diapers.", {"recent_events"}, evidence) == ""
    problem = agent.check_answer("When was the last feed?", "She had 5 wet diapers.", {"recent_events"}, evidence)
    assert "5" in problem


def test_checker_requires_the_right_source():
    assert "search_documents" in agent.check_answer("What should I do about a rash?", "Use cream.", set(), "")
    assert "clinic_visits" in agent.check_answer("What did the nurse say?", "Nothing.", set(), "")
    assert "immunizations" in agent.check_answer("Which vaccines has she had?", "None.", set(), "")


def test_checker_catches_tool_calls_written_as_text():
    answer = 'Let me check. {"name": "log", "parameters": {"type": "feed"}}'
    assert "does not exist" in agent.check_answer("hi", answer, set(), "")


def test_calculate_is_safe():
    assert agent.calculate(None, TZ, "(3400-2950)/3400*100")["result"] == 13.24
    assert "error" in agent.calculate(None, TZ, "__import__('os').system('ls')")
    assert "error" in agent.calculate(None, TZ, "result[0]['feeds']")


def test_totals_question_must_use_the_total():
    evidence = '{"what": "feed: formula, 75 ml"} {"total_ml": 440, "feeds": 7}'
    totals = agent._numbers('{"total_ml": 440, "feeds": 7}')
    used = {"recent_events", "summarize_period"}
    q = "How much did baby drink today?"
    assert "totals" in agent.check_answer(q, "Baby drank 75 ml today.", used, evidence, totals)
    assert agent.check_answer(q, "Baby drank 440 ml today.", used, evidence, totals) == ""

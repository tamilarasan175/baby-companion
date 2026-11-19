import asyncio

from app.models import Event
from app.services import voice_agent as va
from app.services.care_tools import to_number


def test_to_number_handles_llm_strings():
    assert to_number("60") == 60
    assert to_number("60 ml") == 60
    assert to_number("4.5") == 4.5
    assert to_number(None, None) is None
    assert to_number("lots", 0) == 0


def test_parse_amounts_and_units():
    assert va.parse_ml("she had 60 milliliters") == 60
    assert va.parse_ml("3 oz of formula") == 90
    assert va.parse_ml("no amount here") is None
    assert va.parse_grams("she weighs 4.2 kg") == 4200
    assert va.parse_grams("weighs 3400 grams") == 3400
    assert va.parse_minutes("nursed for 15 minutes") == 15
    assert va.parse_brand("some infamil formula") == "Enfamil"


def test_normalize_number_words():
    assert va.normalize("Three ounces") == "3 ounces"


def test_ground_rejects_invented_diaper_and_fixes_amounts():
    t = va.normalize("Baby had 60 milliliters of formula and a wet diaper.")
    # The model invented a dirty diaper and a 1800 ml amount and a brand: code corrects it.
    assert va.ground("log_diaper", {"status": "dirty"}, t) is None
    feed = va.ground("log_feed", {"milk_type": "formula", "ml": 1800, "formula_name": "Similac"}, t)
    assert feed["ml"] == 60
    assert feed["formula_name"] is None
    assert va.ground("log_diaper", {"status": "wet"}, t) == {"status": "wet"}


def test_rule_based_fallback():
    actions = va.rule_based_actions(va.normalize("2 oz of Similac and a wet diaper, then she fell asleep"))
    names = [a[0] for a in actions]
    assert names.count("log_feed") == 1
    assert ("log_diaper", {"status": "wet"}) in actions
    assert ("log_sleep", {"action": "start"}) in actions


def test_keywords_fill_in_an_event_the_model_missed(db, monkeypatch):
    # The model returns only the dirty diaper for "wet and dirty"; the wet one must still be logged.
    async def fake_chat(*args, **kwargs):
        return {"tool_calls": [{"function": {"name": "log_diaper", "arguments": {"status": "dirty"}}}]}
    monkeypatch.setattr(va.llm, "chat", fake_chat)
    result = asyncio.run(va.run(db, "Diaper was wet and dirty"))
    kinds = sorted(e.data["type"] for e in db.query(Event).filter(Event.type == "diaper"))
    assert kinds == ["dirty", "wet"]
    assert result["mode"] == "llm+rules"
    assert result["llm_error"] is None


def test_run_without_llm_uses_rules_and_never_loses_input(db):
    result = asyncio.run(va.run(db, "Baby had 2 oz of Similac and a wet diaper"))
    assert result["mode"] == "rules"
    feeds = db.query(Event).filter(Event.type == "feed").all()
    assert feeds[0].data["amount_ml"] == 60 and feeds[0].data["type"] == "formula"

    result = asyncio.run(va.run(db, "she is coughing a lot"))
    assert result["mode"] == "note"
    assert db.query(Event).filter(Event.type == "note").one().data["text"] == "she is coughing a lot"

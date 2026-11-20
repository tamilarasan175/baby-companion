from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.models import Event
from app.services import agent, baby, growth

TZ = ZoneInfo("America/Edmonton")


def test_who_medians_are_the_50th_percentile():
    # WHO medians at birth: girls 3.2322 kg, boys 3.3464 kg; length girls 49.1477 cm.
    assert round(growth.percentile(growth.z_score("weight", "girl", 0, 3.2322))) == 50
    assert round(growth.percentile(growth.z_score("weight", "boy", 0, 3.3464))) == 50
    assert round(growth.percentile(growth.z_score("length", "girl", 0, 49.1477))) == 50


@pytest.mark.parametrize("sex, kg, expected", [
    # WHO weight-for-age percentile tables at birth (rounded to 0.1 kg there, so allow a few points).
    ("girl", 2.4, 3), ("girl", 4.2, 97), ("boy", 2.5, 3), ("boy", 4.3, 97)])
def test_matches_published_who_percentiles(sex, kg, expected):
    assert abs(growth.percentile(growth.z_score("weight", sex, 0, kg)) - expected) < 2


def test_percentile_text():
    assert growth.percentile_text(45.4) == "45th percentile"
    assert growth.percentile_text(1.2) == "1st percentile"
    assert growth.percentile_text(12) == "12th percentile"
    assert growth.percentile_text(22) == "22nd percentile"
    assert growth.percentile_text(0.4) == "below the 1st percentile"


def _weigh(db, days_after_birth, grams, born):
    ts = datetime.combine(born + timedelta(days=days_after_birth), time(11), tzinfo=TZ)
    db.add(Event(type="weight", data={"grams": grams}, timestamp=ts.astimezone(timezone.utc).replace(tzinfo=None)))


def test_growth_report_needs_sex(db):
    baby.save_profile(db, {"birth_date": "2026-11-01", "birth_weight_g": 3350})
    assert "sex" in growth.growth_report(db, TZ)["error"]


def test_growth_report_percent_of_birth_weight(db):
    born = date.today() - timedelta(days=20)
    baby.save_profile(db, {"sex": "girl", "birth_date": born.isoformat(), "birth_weight_g": 3350})
    for d, g in ((3, 3150), (9, 3350), (16, 3600)):
        _weigh(db, d, g, born)
    db.commit()
    r = growth.growth_report(db, TZ)
    assert [w["source"] for w in r["weight"]] == ["birth", "home scale", "home scale", "home scale"]
    assert r["weight"][0]["percentile"] == "60th percentile"  # 3350 g vs the WHO girls' median of 3232 g
    v = r["vs_birth_weight"]
    assert "-6.0%" in v["lowest"] and "day 3" in v["lowest"]
    assert "+7.5%" in v["latest"]
    assert "day 9" in v["back_to_birth_weight"]


def test_invalid_sex_rejected(db):
    with pytest.raises(ValueError):
        baby.save_profile(db, {"sex": "unknown"})


def test_growth_questions_route_to_the_growth_tool():
    for q in ("What percentile is her weight?", "Is her weight percentile normal?",
              "How much % of birth weight has she gained?", "How big is she compared to other babies?",
              "How much has she gained compared to her birth weight in percent?"):
        assert ({"growth_percentiles"}, "growth_percentiles") in agent.required_sources(q)["logs"], q
    # Plain weight questions accept either weight tool; feed comparisons are untouched.
    assert agent.required_sources("How much weight did she gain?")["logs"][0][0] >= {"weight_history"}
    assert all("growth_percentiles" not in s for _, s in agent.required_sources("Did she eat less today vs yesterday?")["logs"])

"""Growth percentiles from the WHO Child Growth Standards, computed in code (never by the LLM).

who_growth.json holds the WHO LMS parameters for each day of age from birth to 730 days,
for girls and boys: weight (kg), length (cm) and head circumference (cm). It ships with the
app, so nothing is looked up online.

Percentile = where a measurement falls among healthy babies of the same age and sex:
the 50th is the middle; the 25th means 25 out of 100 babies measure less.
"""
import json
import math
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session

from ..models import Event
from . import baby
from .timeutil import to_local

MAX_AGE_DAYS = 730
INDICATORS = {"weight": "weight_kg", "length": "length_cm", "head": "head_cm"}


@lru_cache(maxsize=1)
def _tables() -> dict:
    return json.loads((Path(__file__).with_name("who_growth.json")).read_text())


def z_score(indicator: str, sex: str, age_days: int, value: float) -> float:
    """WHO z-score for a measurement (value in kg for weight, cm otherwise)."""
    L, M, S = _tables()[INDICATORS[indicator]][sex][max(0, min(MAX_AGE_DAYS, age_days))]
    z = math.log(value / M) / S if L == 0 else ((value / M) ** L - 1) / (L * S)
    if indicator == "weight" and abs(z) > 3:
        # WHO's adjustment for weight beyond +/-3 SD (the curve is skewed there).
        def sd(n):
            return M * (1 + L * S * n) ** (1 / L)
        if z > 3:
            z = 3 + (value - sd(3)) / (sd(3) - sd(2))
        else:
            z = -3 + (value - sd(-3)) / (sd(-2) - sd(-3))
    return z


def percentile(z: float) -> float:
    return 50 * (1 + math.erf(z / math.sqrt(2)))


def percentile_text(p: float) -> str:
    if p < 1:
        return "below the 1st percentile"
    if p > 99:
        return "above the 99th percentile"
    n = round(p)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} percentile"


def _measurements(db: Session, profile: dict, tz) -> list:
    """(local date, indicator, value in kg/cm, source) for birth, logged weights and clinic visits."""
    born = date.fromisoformat(profile["birth_date"])
    out = []
    for key, ind, scale in (("birth_weight_g", "weight", 1000), ("birth_length_cm", "length", 1),
                            ("birth_head_cm", "head", 1)):
        if profile.get(key):
            out.append((born, ind, float(profile[key]) / scale, "birth"))
    for e in db.query(Event).filter(Event.type.in_(("weight", "visit"))).order_by(Event.timestamp).all():
        d, day = e.data or {}, to_local(e.timestamp, tz).date()
        if e.type == "weight" and d.get("grams"):
            out.append((day, "weight", float(d["grams"]) / 1000, "home scale"))
        elif e.type == "visit":
            plan = d.get("plan") or {}
            for key, ind, scale in (("weight_g", "weight", 1000), ("length_cm", "length", 1),
                                    ("head_circumference_cm", "head", 1)):
                try:
                    value = float(plan.get(key) or 0) / scale
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    out.append((day, ind, value, "clinic"))
    return out


def growth_report(db: Session, tz) -> dict:
    profile = baby.get_profile(db)
    missing = [label for key, label in (("sex", "sex"), ("birth_date", "birth date")) if not profile.get(key)]
    if missing:
        return {"error": f"Growth percentiles need the baby's {' and '.join(missing)}. "
                         "The parent can add this on the Baby tab."}
    sex, born = profile["sex"], date.fromisoformat(profile["birth_date"])
    unit = {"weight": "g", "length": "cm", "head": "cm"}
    result = {"chart": f"WHO Child Growth Standards ({sex}s)"}
    for ind in ("weight", "length", "head"):
        rows = []
        for day, i, value, source in _measurements(db, profile, tz):
            age = (day - born).days
            if i != ind or not 0 <= age <= MAX_AGE_DAYS:
                continue
            p = percentile(z_score(ind, sex, age, value))
            rows.append({"date": day.strftime("%b %d"), "age_days": age,
                         ind: round(value * 1000) if ind == "weight" else value, "unit": unit[ind],
                         "percentile": percentile_text(p), "source": source})
        if rows:
            result[ind] = rows
    if "weight" not in result:
        result["weight"] = "No weights recorded yet."

    # Percent of birth weight: newborns usually lose some weight, then regain it.
    bw = profile.get("birth_weight_g")
    weights = result["weight"] if isinstance(result["weight"], list) else []
    later = [w for w in weights if w["source"] != "birth"]
    if bw and later:
        bw = float(bw)
        low = min(later, key=lambda w: w["weight"])
        latest = later[-1]
        result["vs_birth_weight"] = {
            "birth_weight_g": round(bw),
            "latest": f"{latest['weight']} g on {latest['date']}, {_pct(latest['weight'], bw)} vs birth weight",
            "lowest": f"{low['weight']} g on {low['date']} (day {low['age_days']}), {_pct(low['weight'], bw)} vs birth weight",
        }
        back = next((w for w in later if w["weight"] >= bw and w["age_days"] >= low["age_days"]), None)
        if back:
            result["vs_birth_weight"]["back_to_birth_weight"] = f"on {back['date']} (day {back['age_days']})"
        elif low["weight"] < bw:
            result["vs_birth_weight"]["back_to_birth_weight"] = "not yet"
    result["how_to_read"] = ("A percentile compares the baby with healthy babies of the same age and sex; "
                             "50th is the middle. Any percentile can be healthy: doctors look at whether the baby "
                             "keeps roughly following the same line over time. Home scales can be off by 50-100 g, "
                             "so clinic measurements are more reliable.")
    return result


def _pct(grams: float, birth: float) -> str:
    change = (grams - birth) / birth * 100
    return f"{change:+.1f}%"

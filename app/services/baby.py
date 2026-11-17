"""Baby profile (birth details, notes) and immunization records."""
from datetime import date, datetime, time, timezone

from sqlalchemy.orm import Session

from ..models import BabyProfile, Event
from .care_tools import to_number


PROFILE_FIELDS = ("name", "birth_date", "birth_time", "birth_weight_g", "birth_length_cm", "birth_head_cm", "notes")
NUMERIC_FIELDS = ("birth_weight_g", "birth_length_cm", "birth_head_cm")


def get_profile(db: Session) -> dict:
    row = db.get(BabyProfile, 1)
    return dict(row.data or {}) if row else {}


def save_profile(db: Session, data: dict) -> dict:
    clean = {}
    for k in PROFILE_FIELDS:
        v = data.get(k)
        if k in NUMERIC_FIELDS:
            v = to_number(v, None)
        elif isinstance(v, str):
            v = v.strip()
        if v not in (None, ""):
            clean[k] = v
    if "birth_date" in clean:
        date.fromisoformat(clean["birth_date"])  # raises ValueError if invalid
    row = db.get(BabyProfile, 1)
    if row:
        row.data = clean
    else:
        db.add(BabyProfile(id=1, data=clean))
    db.commit()
    return clean


def age_text(birth_date: str, tz) -> str:
    """'3 weeks 2 days old (23 days)' style text, or '' if unknown."""
    try:
        born = date.fromisoformat(birth_date)
    except (TypeError, ValueError):
        return ""
    days = (datetime.now(tz).date() - born).days
    if days < 0:
        return "not born yet"
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} old"
    if days < 90:
        return f"{days // 7} weeks {days % 7} days old ({days} days)"
    months = days // 30
    return f"about {months} months old ({days} days)"


def profile_summary(db: Session, tz) -> str:
    """One paragraph for the agent's instructions."""
    p = get_profile(db)
    if not p:
        return "The parent has not filled in the baby profile yet (no name, birth date or birth weight)."
    parts = [f"Baby's name: {p['name']}." if p.get("name") else ""]
    if p.get("birth_date"):
        parts.append(f"Born {p['birth_date']}" + (f" at {p['birth_time']}" if p.get("birth_time") else "") +
                     f", now {age_text(p['birth_date'], tz)}.")
    else:
        parts.append("Birth date unknown.")
    birth = [f"weight {p['birth_weight_g']} g" if p.get("birth_weight_g") else "",
             f"length {p['birth_length_cm']} cm" if p.get("birth_length_cm") else "",
             f"head {p['birth_head_cm']} cm" if p.get("birth_head_cm") else ""]
    if any(birth):
        parts.append("At birth: " + ", ".join(b for b in birth if b) + ".")
    if p.get("notes"):
        parts.append(f"Parent's notes: {p['notes'][:500]}")
    return " ".join(x for x in parts if x)


# ---------------------------------------------------------------- immunizations

def add_immunization(db: Session, name: str, given_on: str, tz, place: str = "", notes: str = "",
                     next_due: str = "") -> Event:
    given = date.fromisoformat(given_on)
    if next_due:
        date.fromisoformat(next_due)
    # Store at local noon so the date never shifts when converted between time zones.
    local_noon = datetime.combine(given, time(12, 0), tzinfo=tz)
    ts = local_noon.astimezone(timezone.utc).replace(tzinfo=None)
    data = {"name": name.strip(), "date": given_on}
    for k, v in (("place", place), ("notes", notes), ("next_due", next_due)):
        if v and v.strip():
            data[k] = v.strip()
    e = Event(type="immunization", data=data, timestamp=ts, source="manual")
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def list_immunizations(db: Session) -> list:
    rows = db.query(Event).filter(Event.type == "immunization").order_by(Event.timestamp).all()
    return [{"id": e.id, **(e.data or {})} for e in rows]


def immunization_report(db: Session, tz) -> dict:
    """Given doses plus any 'next due' dates still ahead, for the agent and the visit brief."""
    given = list_immunizations(db)
    today = datetime.now(tz).date()
    upcoming = []
    for g in given:
        if g.get("next_due"):
            try:
                due = date.fromisoformat(g["next_due"])
            except ValueError:
                continue
            if due >= today:
                upcoming.append({"after": g["name"], "due": g["next_due"], "in_days": (due - today).days})
    upcoming.sort(key=lambda u: u["due"])
    return {"given": [{k: v for k, v in g.items() if k != "id"} for g in given] or "No immunizations recorded yet.",
            "upcoming": upcoming or "No upcoming due dates recorded."}

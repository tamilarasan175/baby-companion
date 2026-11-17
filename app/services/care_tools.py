"""Shared care tools used by voice agent, care coordinator, and MCP."""
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..models import Event


def to_number(value, default=0):
    """LLMs often return numbers as strings ("60", "60 ml"). Totals break if those get stored."""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    try:
        f = float(str(value).strip().split()[0].replace(",", ""))
        return int(f) if f.is_integer() else f
    except (ValueError, IndexError):
        return default


def _save(db: Session, type_: str, data: dict, source: str) -> Event:
    e = Event(type=type_, data=data, source=source)
    db.add(e)
    db.commit()
    return e


def log_feed(db: Session, milk_type: str, ml=0, formula_name: str = None, duration_min=None,
             note: str = None, source: str = "voice"):
    milk_type = "formula" if str(milk_type).lower().startswith("form") else "breast"
    ml = to_number(ml)
    data = {"type": milk_type, "amount_ml": ml}
    if formula_name:
        data["formula_name"] = formula_name
    duration_min = to_number(duration_min, None)
    if duration_min:
        data["duration_sec"] = int(duration_min * 60)
    if note:
        data["note"] = note
    _save(db, "feed", data, source)
    label = "Mom's milk" if milk_type == "breast" else "Formula"
    return {"logged": "feed", "text": f"{label} {ml} ml" if ml else f"{label} feed"}


def log_diaper(db: Session, status: str, note: str = None, source: str = "voice"):
    s = str(status).lower()
    status = "dirty" if any(w in s for w in ("dirty", "poo", "stool", "bm", "soiled")) else "wet"
    data = {"type": status}
    if note:
        data["note"] = note
    _save(db, "diaper", data, source)
    return {"logged": "diaper", "text": f"{status} diaper"}


def log_sleep(db: Session, action: str, source: str = "voice"):
    action = "end" if str(action).lower() in ("end", "stop", "woke", "wake", "awake") else "start"
    data = {"action": action}
    if action == "end":
        last_start = (db.query(Event).filter(Event.type == "sleep")
                      .order_by(Event.timestamp.desc()).first())
        if last_start and (last_start.data or {}).get("action") == "start":
            data["duration_sec"] = int((datetime.utcnow() - last_start.timestamp).total_seconds())
    _save(db, "sleep", data, source)
    mins = data.get("duration_sec", 0) // 60
    return {"logged": "sleep", "text": f"sleep {action}" + (f" ({mins} min)" if mins else "")}


def log_weight(db: Session, grams, source: str = "voice"):
    grams = to_number(grams)
    if not grams:
        return {"logged": None, "text": "weight not understood"}
    _save(db, "weight", {"grams": grams}, source)
    return {"logged": "weight", "text": f"weight {grams} g"}


def log_note(db: Session, text: str, source: str = "voice"):
    _save(db, "note", {"text": text}, source)
    return {"logged": "note", "text": f"note: {text[:60]}"}


def set_reminder(label: str, delay_minutes: int):
    # For now just return the intent; full scheduling can be added later
    return {"reminder": label, "in_minutes": to_number(delay_minutes, 120),
            "text": f"reminder '{label}' noted (not scheduled yet)"}


def get_recent_context(db: Session, hours: int = 24):
    since = datetime.utcnow() - timedelta(hours=hours)
    events = db.query(Event).filter(Event.timestamp >= since).order_by(Event.timestamp.desc()).all()
    feeds = [e for e in events if e.type == "feed"]
    diapers = [e for e in events if e.type == "diaper"]
    return {
        "feeds": len(feeds),
        "wet": len([d for d in diapers if (d.data or {}).get("type") == "wet"]),
        "dirty": len([d for d in diapers if (d.data or {}).get("type") == "dirty"]),
        "notes": [(e.data or {}).get("note") or (e.data or {}).get("text") for e in events if e.type == "note"][:5]
    }

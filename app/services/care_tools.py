"""Shared care tools used by voice agent, care coordinator, and MCP."""
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..models import Event

def log_feed(db: Session, milk_type: str, ml: int = 0, formula_name: str = None, duration_min: float = None, note: str = None):
    data = {"type": milk_type, "amount_ml": ml}
    if formula_name:
        data["formula_name"] = formula_name
    if duration_min:
        data["duration_sec"] = int(duration_min * 60)
    if note:
        data["note"] = note
    e = Event(type="feed", data=data, source="voice")
    db.add(e)
    db.commit()
    return {"logged": "feed", "ml": ml, "type": milk_type}

def log_diaper(db: Session, status: str, note: str = None):
    data = {"type": status}
    if note:
        data["note"] = note
    e = Event(type="diaper", data=data, source="voice")
    db.add(e)
    db.commit()
    return {"logged": "diaper", "status": status}

def set_reminder(label: str, delay_minutes: int):
    # For now just return the intent; full scheduling can be added later
    return {"reminder": label, "in_minutes": delay_minutes}

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

from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..models import SessionLocal, Event

mcp = FastMCP("Baby Companion")

def _db():
    return SessionLocal()

@mcp.tool()
def log_feed(milk_type: str = "breast", ml: int = 0, formula_name: str = "", duration_min: float = 0) -> str:
    """Log a milk feed. milk_type is 'breast' or 'formula'."""
    db = _db()
    try:
        data = {"type": milk_type, "amount_ml": ml}
        if formula_name:
            data["formula_name"] = formula_name
        if duration_min:
            data["duration_sec"] = int(duration_min * 60)
        e = Event(type="feed", data=data, source="mcp")
        db.add(e)
        db.commit()
        return f"Logged {milk_type} {ml} ml"
    finally:
        db.close()

@mcp.tool()
def log_diaper(status: str = "wet", note: str = "") -> str:
    """Log a diaper change. status is 'wet' or 'dirty'."""
    db = _db()
    try:
        e = Event(type="diaper", data={"type": status, "note": note}, source="mcp")
        db.add(e)
        db.commit()
        return f"Logged {status} diaper"
    finally:
        db.close()

@mcp.tool()
def get_visit_brief() -> str:
    """Get a short 72-hour summary of feeds, diapers, and milk."""
    db = _db()
    try:
        now = datetime.utcnow()
        start = now - timedelta(hours=72)
        events = db.query(Event).filter(Event.timestamp >= start).all()
        feeds = [e for e in events if e.type == "feed"]
        milk = sum((e.data or {}).get("amount_ml", 0) for e in feeds)
        wet = sum(1 for e in events if e.type == "diaper" and (e.data or {}).get("type") == "wet")
        dirty = sum(1 for e in events if e.type == "diaper" and (e.data or {}).get("type") == "dirty")
        return f"Last 72h: {len(feeds)} feeds, {milk} ml milk, {wet} wet, {dirty} dirty"
    finally:
        db.close()

@mcp.resource("baby://logs/72h")
def logs_72h() -> str:
    """Last 72 hours of events as plain text."""
    db = _db()
    try:
        start = datetime.utcnow() - timedelta(hours=72)
        events = db.query(Event).filter(Event.timestamp >= start).order_by(Event.timestamp).all()
        lines = []
        for e in events:
            lines.append(f"{e.timestamp.isoformat()} {e.type} {e.data}")
        return "\n".join(lines) or "No events"
    finally:
        db.close()

@mcp.prompt()
def prepare_pediatrician_questions() -> str:
    """Draft questions for a public health nurse or pediatrician visit."""
    return "Look at the baby's recent logs and draft 5 clear questions for a Calgary public health nurse."

"""Ask Baby Companion: a tool-using agent over the baby's logs and the uploaded documents.

How it works (the "agent loop"):
  1. Send the question plus a list of tools to the local LLM.
  2. If the model asks for a tool, run it here in Python and send the result back.
  3. Repeat until the model answers without asking for a tool (or MAX_STEPS is hit).

Every tool is read-only, and all counting and arithmetic happen in code. The
model only decides which tools to call and explains the results. Each step is
yielded as an event so the page can show the agent's work live.
"""
import ast
import json
import operator
import re
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..models import Event
from . import baby, llm, rag
from ..config import EMERGENCY_NUMBER, HEALTH_LINE, HEALTH_LINE_NUMBER
from .timeutil import get_tz, local_midnight_utc, to_local

MAX_STEPS = 8
MAX_RETRIES = 2  # times the checker may send an answer back
MAX_RESULT_CHARS = 6000

SYSTEM_PROMPT = """You are Baby Companion, a careful assistant for a parent of a newborn. You answer questions by calling tools that read the baby's logs and search the parent's uploaded health documents.

Current local time: {now} ({tz}).
About the baby: {profile}

How to work:
- First decide what information you need, then call tools to get it. You may call several tools, one after another, and use earlier results to choose the next tool.
- "What should I do", "is this normal", "is it safe" and how-to questions: call search_documents and answer from the guidance.
- Questions about what the baby did (how much, how many, when, today, this week): call the log tools.
- Questions like "enough wet diapers?" or "is her weight on track?" need both: the baby's numbers AND the guidance.
- Questions about what the doctor, nurse or clinic said: call clinic_visits.
- Use only what "About the baby" and tool results say about her age, birth weight and history. If something is not there, say you don't know. Never assume.
- Questions about vaccines, shots or injections (e.g. RSV): call immunizations.
- Only call tools through the tool interface. Never write a tool call as text.
- To compare two periods (today vs yesterday, this week vs last week), use compare_periods. Mention both feed count and milk volume.
- Every number you state must come from a tool result. Never estimate or invent numbers. Use calculate for any other arithmetic, with plain numbers only.
- Only call something normal, on track, or concerning if a search_documents passage supports it, and cite it as (document title, p. N). If the documents do not cover it, give the facts and say the documents don't say what is normal.
- Read tool results carefully and report exactly what they say. If a result does not answer the question, call another tool.
- You are not a doctor. If something sounds urgent (trouble breathing, fever in a baby under 3 months, very few wet diapers, baby very sleepy and not feeding), tell the parent to call {health_line}, or {emergency} in an emergency.
- Reply in a few short, plain sentences or bullets."""

TOOLS = [
    {"type": "function", "function": {
        "name": "summarize_period",
        "description": "Totals for a time window: number of feeds, breast milk ml, formula ml, wet and dirty diapers, sleep minutes, and daily averages. "
                       "days = window length. offset_days = how many days before now the window ends (0 = ends now). "
                       "Example: this week = days 7, offset 0; last week = days 7, offset 7.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "Window length in days (1-60)"},
            "offset_days": {"type": "integer", "description": "Days before now that the window ends; 0 means now"}},
            "required": ["days"]}}},
    {"type": "function", "function": {
        "name": "compare_periods",
        "description": "Compare the most recent N days with the N days before that (e.g. days=1: last 24h vs the 24h before; days=7: this week vs last week). "
                       "Returns both totals and the change in percent for feeds, milk ml, diapers and sleep.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "Length of each period in days"}},
            "required": ["days"]}}},
    {"type": "function", "function": {
        "name": "daily_breakdown",
        "description": "One row per calendar day (local time) with feeds, ml, diapers and sleep. Good for spotting trends.",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "How many recent days, including today (1-31)"}},
            "required": ["days"]}}},
    {"type": "function", "function": {
        "name": "recent_events",
        "description": "The most recent individual log entries, newest first, with time and how long ago. Use for 'when was the last...' questions.",
        "parameters": {"type": "object", "properties": {
            "event_type": {"type": "string", "enum": ["any", "feed", "diaper", "wet_diaper", "dirty_diaper", "sleep", "weight", "note", "visit"]},
            "limit": {"type": "integer", "description": "Max entries (1-20)"}},
            "required": ["event_type"]}}},
    {"type": "function", "function": {
        "name": "weight_history",
        "description": "All logged weights with dates, plus overall gain, average grams gained per day and per week.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the parent's uploaded health documents (e.g. public health guides). Use for any 'what should I do', 'is this normal' or how-to question. Returns matching passages with title and page.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A few key words, e.g. 'wet diapers per day'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "clinic_visits",
        "description": "What the doctor, nurse or clinic said at recorded visits, newest first: date, days ago, summary, weight, feeding plan, medications, warning signs, next visit, and part of the transcript. Returns only the latest visit unless count is higher.",
        "parameters": {"type": "object", "properties": {
            "count": {"type": "integer", "description": "How many recent visits (1-3). Default 1 = the latest visit."}}}}},
    {"type": "function", "function": {
        "name": "immunizations",
        "description": "Vaccines and injections the baby has had (name, date, where, notes) and any recorded next due dates.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate arithmetic with plain numbers only (no names), e.g. '(3400-2950)/3400*100'. Supports + - * / ( ) and round().",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}},
            "required": ["expression"]}}},
]


# ---------------------------------------------------------------- tools

def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _totals(events: list) -> dict:
    feeds = [e for e in events if e.type == "feed"]
    ml = lambda kind: sum(int((e.data or {}).get("amount_ml") or 0) for e in feeds if (e.data or {}).get("type") == kind)
    breast, formula = ml("breast"), ml("formula")
    diapers = [e for e in events if e.type == "diaper"]
    sleep_sec = sum(int((e.data or {}).get("duration_sec") or 0) for e in events
                    if e.type == "sleep" and (e.data or {}).get("action") == "end")
    fed_ml = [int((e.data or {}).get("amount_ml") or 0) for e in feeds if (e.data or {}).get("amount_ml")]
    return {
        "feeds": len(feeds),
        "breast_ml": breast,
        "formula_ml": formula,
        "total_ml": breast + formula,
        "avg_ml_per_feed": round(sum(fed_ml) / len(fed_ml)) if fed_ml else None,
        "wet_diapers": sum(1 for e in diapers if (e.data or {}).get("type") == "wet"),
        "dirty_diapers": sum(1 for e in diapers if (e.data or {}).get("type") == "dirty"),
        "sleep_minutes": round(sleep_sec / 60),
    }


def summarize_period(db, tz, days=1, offset_days=0):
    days, offset_days = _clamp(days, 1, 60, 1), _clamp(offset_days, 0, 365, 0)
    end = datetime.utcnow() - timedelta(days=offset_days)
    start = end - timedelta(days=days)
    events = db.query(Event).filter(Event.timestamp >= start, Event.timestamp < end).all()
    totals = _totals(events)
    per_day = {f"{k}_per_day": round(v / days, 1) for k, v in totals.items()
               if k in ("feeds", "total_ml", "wet_diapers", "dirty_diapers", "sleep_minutes")}
    fmt = "%b %d %H:%M"
    return {"window": f"{to_local(start, tz).strftime(fmt)} to {to_local(end, tz).strftime(fmt)} ({days} days)",
            **totals, **per_day}


def compare_periods(db, tz, days=7):
    days = _clamp(days, 1, 30, 7)
    recent = summarize_period(db, tz, days, 0)
    before = summarize_period(db, tz, days, days)
    change = {}
    for k in ("feeds", "total_ml", "breast_ml", "formula_ml", "wet_diapers", "dirty_diapers", "sleep_minutes"):
        a, b = recent[k], before[k]
        change[k] = f"{a - b:+d} ({round((a - b) / b * 100):+d}%)" if b else f"{a - b:+d}"
    return {"recent_period": recent, "previous_period": before, "change_recent_vs_previous": change}


def daily_breakdown(db, tz, days=7):
    days = _clamp(days, 1, 31, 7)
    rows = []
    for i in range(days):
        start, end = local_midnight_utc(tz, i), local_midnight_utc(tz, i - 1)
        events = db.query(Event).filter(Event.timestamp >= start, Event.timestamp < end).all()
        label = to_local(start, tz).strftime("%a %b %d") + (" (today, so far)" if i == 0 else "")
        t = _totals(events)
        rows.append({"day": label, "feeds": t["feeds"], "total_ml": t["total_ml"],
                     "wet": t["wet_diapers"], "dirty": t["dirty_diapers"], "sleep_min": t["sleep_minutes"]})
    return {"days": rows}


def _describe(e: Event) -> str:
    d = e.data or {}
    if e.type == "feed":
        kind = "breast milk" if d.get("type") == "breast" else "formula"
        parts = [kind]
        if d.get("amount_ml"):
            parts.append(f"{d['amount_ml']} ml")
        if d.get("duration_sec"):
            parts.append(f"{round(d['duration_sec'] / 60)} min")
        if d.get("formula_name"):
            parts.append(d["formula_name"])
        return "feed: " + ", ".join(parts)
    if e.type == "diaper":
        return f"{d.get('type', '')} diaper"
    if e.type == "sleep":
        mins = f" after {round(d['duration_sec'] / 60)} min" if d.get("duration_sec") else ""
        return "fell asleep" if d.get("action") == "start" else f"woke up{mins}"
    if e.type == "weight":
        return f"weight {d.get('grams')} g"
    if e.type == "note":
        return f"note: {d.get('text', '')}"
    if e.type == "visit":
        return "clinic visit: " + ((d.get("plan") or {}).get("summary") or "")
    return e.type


def _ago(ts: datetime) -> str:
    mins = int((datetime.utcnow() - ts).total_seconds() // 60)
    if mins < 60:
        return f"{mins} min ago"
    if mins < 48 * 60:
        return f"{mins // 60} h {mins % 60} min ago"
    return f"{mins // 1440} days ago"


def recent_events(db, tz, event_type="any", limit=10):
    limit = _clamp(limit, 1, 20, 10)
    q = db.query(Event).filter(Event.type != "care_card")
    diaper_kind = {"wet_diaper": "wet", "dirty_diaper": "dirty"}.get(event_type)
    if diaper_kind:
        q = q.filter(Event.type == "diaper")
    elif event_type and event_type != "any":
        q = q.filter(Event.type == event_type)
    events = q.order_by(Event.timestamp.desc()).limit(200 if diaper_kind else limit).all()
    if diaper_kind:
        events = [e for e in events if (e.data or {}).get("type") == diaper_kind][:limit]
    return {"events": [{"time": to_local(e.timestamp, tz).strftime("%a %b %d %H:%M"), "ago": _ago(e.timestamp),
                        "what": _describe(e)} for e in events] or "No matching entries."}


def weight_history(db, tz):
    # Points as (utc naive timestamp, grams, label); the birth weight from the profile comes first.
    points = []
    profile = baby.get_profile(db)
    if profile.get("birth_weight_g") and profile.get("birth_date"):
        born = datetime.fromisoformat(profile["birth_date"] + "T" + (profile.get("birth_time") or "12:00"))
        born_utc = born.replace(tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)
        points.append((born_utc, profile["birth_weight_g"], "birth weight"))
    for e in db.query(Event).filter(Event.type == "weight").order_by(Event.timestamp).all():
        if (e.data or {}).get("grams"):
            points.append((e.timestamp, e.data["grams"], ""))
    if not points:
        return {"weights": "No weights logged yet."}
    points.sort(key=lambda p: p[0])
    rows, prev = [], None
    for ts, grams, label in points:
        row = {"date": to_local(ts, tz).strftime("%b %d"), "grams": grams}
        if label:
            row["note"] = label
        if prev:
            days = max((ts - prev[0]).total_seconds() / 86400, 0.01)
            row["gain_since_previous_g"] = grams - prev[1]
            row["gain_per_day_g"] = round((grams - prev[1]) / days)
        rows.append(row)
        prev = (ts, grams)
    result = {"weights": rows}
    if len(points) >= 2:
        (t0, g0, l0), (t1, g1, _) = points[0], points[-1]
        days = max((t1 - t0).total_seconds() / 86400, 0.01)
        gain = g1 - g0
        result["overall"] = {"from": f"{g0} g on {to_local(t0, tz).strftime('%b %d')}" + (f" ({l0})" if l0 else ""),
                             "to": f"{g1} g on {to_local(t1, tz).strftime('%b %d')}",
                             "total_gain_g": gain, "days": round(days, 1),
                             "avg_gain_per_day_g": round(gain / days), "avg_gain_per_week_g": round(gain / days * 7)}
    return result


def immunizations(db, tz):
    return baby.immunization_report(db, tz)


def search_documents(db, tz, query=""):
    hits = rag.search(db, query or "", k=4)
    if not hits:
        return {"passages": "No matching passages in the uploaded documents."}
    return {"passages": [{"title": title, "page": c.page, "text": c.content[:700]} for _, c, title in hits]}


def clinic_visits(db, tz, count=1, limit=None):
    count = _clamp(limit if limit is not None else count, 1, 3, 1)
    visits = db.query(Event).filter(Event.type == "visit").order_by(Event.timestamp.desc()).limit(count).all()
    if not visits:
        return {"visits": "No clinic visits recorded yet."}
    out = []
    today = datetime.now(tz).date()
    for i, v in enumerate(visits):
        local = to_local(v.timestamp, tz)
        days_ago = (today - local.date()).days
        when = "today" if days_ago == 0 else "yesterday" if days_ago == 1 else f"{days_ago} days ago"
        plan = {k: val for k, val in ((v.data or {}).get("plan") or {}).items() if val not in (None, "")}
        transcript = (v.data or {}).get("transcript") or ""
        limit = 1500 if i == 0 else 500
        out.append({"date": local.strftime("%a %b %d %H:%M"), "when": when, **plan,
                    "transcript_excerpt": transcript[:limit] + ("..." if len(transcript) > limit else "")})
    return {"visits": out}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("round", "abs") and len(node.args) <= 2:
        fn = round if node.func.id == "round" else abs
        return fn(*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculate(db, tz, expression=""):
    try:
        value = _eval(ast.parse(str(expression).replace("×", "*").replace("÷", "/"), mode="eval"))
        return {"expression": expression, "result": round(value, 2) if isinstance(value, float) else value}
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
        return {"error": f"Could not calculate '{expression}': {e}"}


TOOL_FUNCS = {
    "summarize_period": summarize_period,
    "compare_periods": compare_periods,
    "daily_breakdown": daily_breakdown,
    "recent_events": recent_events,
    "weight_history": weight_history,
    "search_documents": search_documents,
    "clinic_visits": clinic_visits,
    "immunizations": immunizations,
    "calculate": calculate,
}


def run_tool(db: Session, tz, name: str, args: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOL_FUNCS)}"}
    allowed = fn.__code__.co_varnames[2:fn.__code__.co_argcount]
    try:
        return fn(db, tz, **{k: v for k, v in (args or {}).items() if k in allowed})
    except Exception as e:  # report to the model so it can try something else
        return {"error": f"{name} failed: {e}"}


def short_summary(name: str, result: dict) -> str:
    """One line for the trace UI."""
    if "error" in result:
        return result["error"]
    if name == "summarize_period":
        return (f"{result['feeds']} feeds, {result['total_ml']} ml, {result['wet_diapers']} wet, "
                f"{result['dirty_diapers']} dirty, {result['sleep_minutes']} min sleep · {result['window']}")
    if name == "compare_periods":
        c = result["change_recent_vs_previous"]
        return f"feeds {c['feeds']}, milk {c['total_ml']} ml, wet {c['wet_diapers']}, dirty {c['dirty_diapers']} (last {result['recent_period']['window'].split('(')[1]}"
    if name == "daily_breakdown":
        return f"{len(result['days'])} days of totals"
    if name == "recent_events":
        ev = result["events"]
        return ev if isinstance(ev, str) else f"{len(ev)} entries, latest: {ev[0]['what']} ({ev[0]['ago']})"
    if name == "weight_history":
        w = result["weights"]
        if isinstance(w, str):
            return w
        o = result.get("overall")
        return f"{len(w)} weights, latest {w[-1]['grams']} g on {w[-1]['date']}" + (f", avg +{o['avg_gain_per_day_g']} g/day" if o else "")
    if name == "search_documents":
        p = result["passages"]
        return p if isinstance(p, str) else "; ".join(f"{x['title']} p.{x['page']}" for x in p)
    if name == "clinic_visits":
        v = result["visits"]
        return v if isinstance(v, str) else f"{len(v)} visit(s), latest {v[0]['date']}"
    if name == "immunizations":
        g, u = result["given"], result["upcoming"]
        given = g if isinstance(g, str) else f"{len(g)} recorded, latest {g[-1]['name']} ({g[-1]['date']})"
        return given + ("" if isinstance(u, str) else f"; next due {u[0]['due']}")
    if name == "calculate":
        return f"{result['expression']} = {result['result']}"
    return ""


# ---------------------------------------------------------------- loop

_EMBEDDED_CALL = re.compile(r'\{\s*"name"\s*:')


def _embedded_call(content: str):
    """Llama sometimes writes a tool call as JSON text instead of making a real tool call.
    Returns (text_before_json, {"name":..., "arguments":...}) or None."""
    m = _EMBEDDED_CALL.search(content or "")
    if not m:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(content[m.start():])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return content[:m.start()].strip(), {"name": obj.get("name"), "arguments": obj.get("parameters") or obj.get("arguments") or {}}


# ---------------------------------------------------------------- safety notice
# Fixed text shown before any AI output when a question mentions a newborn warning sign.
# Deliberately not generated by the model.
RED_FLAGS = re.compile(
    r"(not|n't|wont|refus\w*|stopped|hard to|can'?t|cannot|unable to)\s+(\w+\s+){0,2}(drink|drinking|feed|feeding|eat|eating|nurse|nursing|latch|wake|waking|breath\w*)|"
    r"(very|too|really|so|extremely) (sleepy|drowsy)|letharg|floppy|\blimp\b|unresponsive|won'?t wake|"
    r"fever|temperature|\bhot\b|breath|blue|grey lips|gray lips|seizure|convuls|"
    r"no wet|dry diaper|not peeing|green vomit|vomit\w* green|blood|dehydrat|jaundice|yellow")
SAFETY_TEXT = (f"This can be a warning sign in a young baby. If it is happening now, call {HEALTH_LINE} for advice. "
               f"Call {EMERGENCY_NUMBER} if your baby is hard to wake, limp, having trouble breathing, or has blue or grey lips.")


# ---------------------------------------------------------------- answer checker
# Small models often answer after one tool call, or state numbers they never looked
# up. Before an answer is shown, code checks it; if it fails, the agent is sent back
# to work with a note saying what's missing ("reflection" step).

LOG_TOOLS = {"summarize_period", "compare_periods", "daily_breakdown", "recent_events"}
TOTAL_TOOLS = {"summarize_period", "compare_periods", "daily_breakdown"}
TOTAL_Q = re.compile(r"how (much|many)|\btotal|altogether|in all|average")
# Question types, decided in code (not by the model):
ADVICE_Q = re.compile(r"what (should|do|can|shall) (i|we) do|what to do|what should|what do the (docs?|documents?|guides?)|"
                      r"\b(docs?|documents?|guides?)\b|is (it|this|that) (normal|ok|okay|safe|bad|serious)|should (i|we)|"
                      r"how (do|can|should) (i|we)|how to\b|\bwhy\b|worr|concern|\bhelp\b|\btips?\b|advice|signs? of|"
                      r"recommend|\bsafe\b|normal|on track|enough")
DATA_Q = re.compile(r"how (much|many|long|often)|when (was|did|is)|\blast\b|today|yesterday|tonight|this (week|morning)|"
                    r"last (week|night)|compare|\bvs\b|versus|trend|total|average|on track|enough|\b(did|has|is) (she|he)\b|"
                    r"\blogs?\b|eating less|eating more")
VACCINE_Q = re.compile(r"vaccin|immuni[sz]|\bshots?\b|injection|\bjabs?\b|needle|\brsv\b|beyfortus|nirsevimab|vitamin k|hep(atitis)? b")
VISIT_Q = re.compile(r"\b(say|said|says|told|tell|mention\w*)\b.*\b(she|he|they|doctor|nurse|dr)\b|\b(she|he|they) (say|said|told|mention\w*)|doctor|\bnurse|clinic|visit|appointment|pediatrician|paediatrician|physician|\bgp\b|midwife|\bdr\b")
# Numbers that appear naturally in answers ("past 24 hours", "call 811").
ALWAYS_OK_NUMBERS = {"24", "48", "72", HEALTH_LINE_NUMBER, EMERGENCY_NUMBER}
TOPICS = [
    (re.compile(r"weigh|weight|gain|grow"), {"weight_history"}, "weight_history"),
    (re.compile(r"diaper|\bwet\b|dirty|poo|pee"), LOG_TOOLS, "compare_periods, summarize_period or recent_events"),
    (re.compile(r"\b(eat|ate|eating|eats|feed|feeds|feeding|fed|milk|ml|drink|drank|bottle)\b"), LOG_TOOLS,
     "compare_periods or summarize_period"),
    (re.compile(r"sleep|slept|\bnap"), LOG_TOOLS, "summarize_period or daily_breakdown"),
]
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set:
    out = set()
    for raw in _NUM.findall(text or ""):
        n = raw.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def required_sources(question: str) -> dict:
    """Which kinds of information a question needs: {'docs': bool, 'visits': bool, 'logs': [(needed_tools, suggestion)]}."""
    q = question.lower()
    advice, data = bool(ADVICE_Q.search(q)), bool(DATA_Q.search(q))
    logs = []
    if data or not advice:  # pure advice questions ("what should I do?") don't need the logs
        logs = [(needed, suggestion) for pattern, needed, suggestion in TOPICS if pattern.search(q)]
        if logs and TOTAL_Q.search(q):  # "how much today?" needs a total, not just the last entry
            logs = [(TOTAL_TOOLS, "daily_breakdown (for today) or summarize_period (for a number of days)")]
    vaccines = bool(VACCINE_Q.search(q))
    if vaccines and not data:  # "is the RSV shot safe?" is advice; "which shots has she had?" is a record lookup
        vaccines = not advice or bool(re.search(r"\b(has|had|got|get|given|did|when|which|next|due)\b", q))
    return {"docs": advice, "visits": bool(VISIT_Q.search(q)), "vaccines": vaccines, "logs": logs}


def check_answer(question: str, answer: str, used_tools: set, evidence: str, total_numbers: set = None) -> str:
    """Return a description of the problem, or '' if the answer passes.

    total_numbers: numbers from totals tools (summarize_period, compare_periods, daily_breakdown).
    """
    embedded = _embedded_call(answer)
    if embedded:
        name = embedded[1]["name"]
        return (f"You wrote a tool call as text{f' using {name!r}, which does not exist' if name not in TOOL_FUNCS else ''}. "
                f"Make a real tool call instead. Tools: {', '.join(TOOL_FUNCS)}.")
    need = required_sources(question)
    if need["vaccines"] and "immunizations" not in used_tools:
        return "This is about her vaccines or injections. Call immunizations first."
    if need["visits"] and "clinic_visits" not in used_tools:
        return "This is about what the doctor or clinic said. Call clinic_visits first."
    if need["docs"] and "search_documents" not in used_tools:
        return "This asks for guidance. Call search_documents first and answer from the documents."
    for needed, suggestion in need["logs"]:
        if not (used_tools & needed):
            if needed == TOTAL_TOOLS and used_tools & LOG_TOOLS:
                return f"This question asks for a total, not single entries. Call {suggestion} first."
            return f"You have not looked at the baby's logs for this. Call {suggestion} first."
    # "How much today?" must be answered with a number from a totals result, not e.g. the last feed's amount.
    if total_numbers and TOTAL_Q.search(question.lower()) and need["logs"]:
        mentioned = {n for n in _numbers(answer) if not (n.isdigit() and int(n) <= 2)}
        if mentioned and not (mentioned & total_numbers):
            return ("Your answer does not use the totals from summarize_period or daily_breakdown. "
                    "Answer with the total from that result.")
    unsupported = sorted(n for n in _numbers(answer) - _numbers(evidence) - ALWAYS_OK_NUMBERS
                         if not (n.isdigit() and int(n) <= 2))
    if unsupported:
        return (f"Your answer mentions {', '.join(unsupported[:5])}, but no tool result contains "
                f"{'that number' if len(unsupported) == 1 else 'those numbers'}. "
                "Call a tool to get the real figure, or use calculate, then answer again.")
    return ""


async def run(db: Session, question: str, history: list = None, tz_name: str = None):
    """Async generator of trace events: status, thought, tool_call, tool_result, check, answer, warning, error, done."""
    tz = get_tz(tz_name)
    started = time.monotonic()
    now = datetime.now(tz).strftime("%A %b %d %Y, %H:%M")
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(now=now, tz=tz.key, profile=baby.profile_summary(db, tz),
                                                             health_line=HEALTH_LINE, emergency=EMERGENCY_NUMBER)}]
    for turn in (history or [])[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])[:1500]})
    messages.append({"role": "user", "content": question})

    # Numbers the answer may legitimately use: the question, the conversation, the clock, and tool results.
    evidence = [question, now, tz.key, messages[0]["content"]] + [str(t.get("content", "")) for t in (history or [])]
    used_tools, retries = set(), 0
    total_numbers = set()
    tools_used = 0
    if RED_FLAGS.search(question.lower()):
        yield {"type": "safety", "text": SAFETY_TEXT}
    need = required_sources(question)
    plan = (["health documents"] if need["docs"] else []) + (["visit notes"] if need["visits"] else []) + \
           (["immunization record"] if need["vaccines"] else []) + \
           (["baby's logs"] if need["logs"] else [])
    if plan:
        yield {"type": "route", "text": "Needs: " + " + ".join(plan)}
        # Tell the model too, so it picks the right tools first time instead of being caught by the checker.
        messages[-1]["content"] += f"\n\n(To answer this, look at: {', '.join(plan)}.)"
    for step in range(1, MAX_STEPS + 1):
        yield {"type": "status", "step": step, "text": "Deciding what to do next..." if step > 1 else "Reading your question..."}
        try:
            msg = await llm.chat(messages, tools=TOOLS, timeout=180.0, temperature=0.1)
        except llm.LLMError as e:
            yield {"type": "error", "text": f"The local AI model is not responding: {e}"}
            return

        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        if not calls:
            embedded = _embedded_call(content)
            if embedded and embedded[1]["name"] in TOOL_FUNCS:
                content, calls = embedded[0], [{"function": embedded[1]}]

        if not calls:
            problem = check_answer(question, content, used_tools, "\n".join(evidence), total_numbers)
            if problem and retries < MAX_RETRIES and step < MAX_STEPS:
                retries += 1
                yield {"type": "check", "ok": False, "text": problem, "draft": content}
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Check failed: {problem}"})
                continue
            if problem:
                yield {"type": "answer", "text": content or "I couldn't come up with an answer."}
                yield {"type": "warning", "text": f"Not fully verified: {problem.split('. Call')[0]}."}
            else:
                yield {"type": "check", "ok": True, "text": "Every number in the answer comes from a tool result."}
                yield {"type": "answer", "text": content or "I couldn't come up with an answer. Try rephrasing the question."}
            yield {"type": "done", "steps": step, "tools_used": tools_used, "seconds": round(time.monotonic() - started)}
            return

        if content:
            yield {"type": "thought", "text": content}
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            name, args = fn.get("name", ""), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            yield {"type": "tool_call", "name": name, "args": args}
            result = run_tool(db, tz, name, args)
            tools_used += 1
            if "error" not in result:
                used_tools.add(name)
                evidence.append(json.dumps(result, default=str))
                if name in TOTAL_TOOLS:
                    total_numbers |= _numbers(json.dumps(result, default=str))
            yield {"type": "tool_result", "name": name, "summary": short_summary(name, result), "data": result}
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, default=str)[:MAX_RESULT_CHARS]})

    # Out of steps: ask for a final answer from what was gathered, with tools disabled.
    yield {"type": "status", "step": MAX_STEPS + 1, "text": "Writing the answer..."}
    messages.append({"role": "user", "content": "Stop calling tools. Answer my question now using only the tool results above."})
    try:
        msg = await llm.chat(messages, timeout=180.0, temperature=0.1)
        final = (msg.get("content") or "").strip() or "I couldn't finish the answer."
        yield {"type": "answer", "text": final}
        problem = check_answer(question, final, used_tools, "\n".join(evidence), total_numbers)
        if problem:
            yield {"type": "warning", "text": f"Not fully verified: {problem.split('. Call')[0]}."}
    except llm.LLMError as e:
        yield {"type": "error", "text": f"The local AI model is not responding: {e}"}
        return
    yield {"type": "done", "steps": MAX_STEPS + 1, "tools_used": tools_used, "seconds": round(time.monotonic() - started)}

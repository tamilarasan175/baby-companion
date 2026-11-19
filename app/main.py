from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone, timedelta
from typing import List
import asyncio
import base64
import json
import secrets
import os
import re
import uuid
import httpx
from fastapi.responses import StreamingResponse
from .models import SessionLocal, Event, Document, Chunk
from .services.care_tools import get_recent_context, to_number
from .services.mcp_server import mcp as mcp_server
from .services import agent, audio, baby, llm, rag, voice_agent
from .services.timeutil import get_tz, to_local, utc_iso
from .config import APP_PASSWORD, DOCS_DIR, HEALTH_LINE, REGION
from .services.llm import OLLAMA_URL, OLLAMA_MODEL


@asynccontextmanager
async def lifespan(app):
    db = SessionLocal()
    try:
        added = rag.ingest_folder(db, DOCS_DIR)
        if added:
            print(f"[rag] ingested from docs/: {', '.join(added)}")
    finally:
        db.close()
    yield


app = FastAPI(title="Baby Companion", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class PasswordMiddleware:
    """Optional login for everything (web UI, API, /mcp) when APP_PASSWORD is set.

    Uses HTTP Basic auth: the browser asks once and remembers it for the session.
    Any username works; only the password is checked. Plain ASGI middleware so
    streaming responses (the agent trace, MCP) are not buffered.
    """
    # Browsers fetch the web-app manifest and icons without credentials.
    OPEN_PATHS = ("/static/manifest.json", "/static/icon-192.png", "/static/icon-512.png")

    def __init__(self, app):
        self.app = app

    def _authorized(self, scope) -> bool:
        for name, value in scope.get("headers", []):
            if name == b"authorization" and value[:6].lower() == b"basic ":
                try:
                    _, _, password = base64.b64decode(value[6:]).decode().partition(":")
                except (ValueError, UnicodeDecodeError):
                    return False
                return secrets.compare_digest(password.encode(), APP_PASSWORD.encode())
        return False

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or not APP_PASSWORD or scope.get("path") in self.OPEN_PATHS
                or self._authorized(scope)):
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"www-authenticate", b'Basic realm="Baby Companion", charset="UTF-8"'),
                                (b"content-type", b"text/plain; charset=utf-8")]})
        await send({"type": "http.response.body", "body": b"Login required"})


app.add_middleware(PasswordMiddleware)

# The page sends the user's IANA time zone (e.g. "America/Edmonton") so "today"
# and dates in answers match the user's clock.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class EventCreate(BaseModel):
    type: str
    data: dict
    source: str = "manual"

class EventOut(BaseModel):
    id: int
    timestamp: datetime
    type: str
    data: dict
    source: str
    model_config = ConfigDict(from_attributes=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/api/voice")
async def voice_transcribe(file: UploadFile = File(...)):
    """Transcribe a short voice note and return text."""
    content = await file.read()
    try:
        transcript = await audio.transcribe(content, file.filename or "voice.webm", prompt=voice_agent.WHISPER_HINT)
    except audio.TranscriptionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"transcript": transcript}


class VoiceAgentIn(BaseModel):
    transcript: str


@app.post("/api/voice-agent")
async def voice_agent_endpoint(body: VoiceAgentIn, db: Session = Depends(get_db)):
    """Turn a transcript into log entries (LLM tool calls, with a rule-based fallback)."""
    return await voice_agent.run(db, body.transcript)

@app.post("/events", response_model=EventOut)
def create_event(event: EventCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_event = Event(type=event.type, data=event.data, source=event.source)
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    # Phase 2: trigger care check in background
    background_tasks.add_task(run_care_check, db_event.id)
    return db_event

async def run_care_check(event_id: int):
    """Lightweight care coordinator. Only alerts on concrete flags."""
    db = SessionLocal()
    try:
        ctx = get_recent_context(db, hours=24)
        prompt = f"""Review these last 24h baby logs. Return ONLY valid JSON: {{"alert_needed": bool, "severity": "info|warn", "message": str, "citation": str}}.
Set alert_needed true ONLY if there is a clear concern (e.g. very few wet diapers, long feed gap with symptoms). Otherwise stay silent.
Logs: {ctx}"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                })
                text = resp.json().get("response", "")
                import json
                # crude extract
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    alert = json.loads(text[start:end])
                    if alert.get("alert_needed"):
                        # store as a note for now
                        e = Event(type="care_card", data=alert, source="agent")
                        db.add(e)
                        db.commit()
        except Exception:
            pass
    finally:
        db.close()

@app.get("/care-cards")
def care_cards(db: Session = Depends(get_db)):
    cards = db.query(Event).filter(Event.type == "care_card").order_by(Event.timestamp.desc()).limit(5).all()
    return [{"id": c.id, "data": c.data, "time": c.timestamp} for c in cards]

@app.get("/events", response_model=List[EventOut])
def list_events(limit: int = 50, db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.timestamp.desc()).limit(limit).all()
    for e in events:
        if e.timestamp and e.timestamp.tzinfo is None:
            e.timestamp = e.timestamp.replace(tzinfo=timezone.utc)
    return events

@app.post("/documents")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(rag.SUPPORTED_EXTS):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT or MD.")
    content = await file.read()
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, filename)
    with open(path, "wb") as f:
        f.write(content)
    try:
        doc = rag.ingest(db, filename, content, path)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Could not read {filename}: {e}")
    chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).count()
    if chunks == 0:
        warning = "No text found. If this is a scanned PDF, it needs OCR before it can be searched."
    else:
        warning = None
    return {"id": doc.id, "title": doc.title, "chars": len(doc.content or ""), "chunks": chunks, "warning": warning}

@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.ingested_at).all()
    return [{"id": d.id, "title": d.title, "ingested_at": d.ingested_at,
             "chunks": db.query(Chunk).filter(Chunk.document_id == d.id).count()} for d in docs]

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
    if doc.source_path and os.path.isfile(doc.source_path):
        os.remove(doc.source_path)
    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}

NOT_FOUND_ANSWER = ("I could not find this in the uploaded documents. "
                    f"Please call {HEALTH_LINE} or check with your healthcare provider.")

RAG_SYSTEM_PROMPT = f"""You are Baby Companion, a baby-care assistant for parents in {REGION}. Answer the parent's question using ONLY the numbered excerpts provided.
- Start directly with the answer, as short clear bullet points. No filler.
- Only include information that answers this specific question. Ignore unrelated sidebars.
- Pay close attention to table headings. Do not mix up opposite columns (e.g. "signs baby is full" vs "hunger cues").
- If only part of a multi-part question is covered, answer that part and say the other part was not found.
- Cite excerpts inline like [1] or [2].
- If the excerpts do not answer the question, reply with exactly: NOT_FOUND"""

@app.get("/api/logs/ask")
def ask_logs(q: str, tz: str = None, db: Session = Depends(get_db)):
    """Answer questions about past logs, including dates (in the user's time zone)."""
    zone = get_tz(tz)
    q_lower = q.lower()
    events = db.query(Event).order_by(Event.timestamp).all()
    for e in events:
        e.local = to_local(e.timestamp, zone)

    # Try to extract a date like "nov 16", "november 8", "11/16"
    target_date = None
    m = re.search(r'(nov|november|oct|october|dec|december|jan|january)\s+(\d{1,2})', q_lower)
    if m:
        month_map = {"jan":1,"january":1,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}
        mon = month_map.get(m.group(1), 11)
        day = int(m.group(2))
        year = datetime.now(zone).year
        try:
            target_date = datetime(year, mon, day)
        except ValueError:
            pass
    elif "today" in q_lower:
        target_date = datetime.now(zone).replace(tzinfo=None)
    elif "yesterday" in q_lower:
        target_date = (datetime.now(zone) - timedelta(days=1)).replace(tzinfo=None)

    def on_day(e, d):
        if not d:
            return True
        return e.local.date() == d.date()

    day_events = [e for e in events if on_day(e, target_date)] if target_date else events

    if "diaper" in q_lower or "wet" in q_lower or "dirty" in q_lower:
        wet = sum(1 for e in day_events if e.type == "diaper" and (e.data or {}).get("type") == "wet")
        dirty = sum(1 for e in day_events if e.type == "diaper" and (e.data or {}).get("type") == "dirty")
        label = target_date.strftime("%b %d") if target_date else "all time"
        return {"answer": f"{label}: {wet} wet, {dirty} dirty diapers."}

    if "weight" in q_lower:
        weights = [e for e in day_events if e.type == "weight"]
        if not weights:
            return {"answer": "No weight found for that day."}
        w = weights[-1]
        return {"answer": f"Weight on {w.local.strftime('%b %d')}: {(w.data or {}).get('grams','?')} g"}

    if "summary" in q_lower or "what happened" in q_lower or "happened" in q_lower:
        if not day_events:
            return {"answer": "No events that day."}
        feeds = [e for e in day_events if e.type == "feed"]
        ml = sum((e.data or {}).get("amount_ml", 0) for e in feeds)
        wet = sum(1 for e in day_events if e.type == "diaper" and (e.data or {}).get("type") == "wet")
        dirty = sum(1 for e in day_events if e.type == "diaper" and (e.data or {}).get("type") == "dirty")
        weights = [e for e in day_events if e.type == "weight"]
        wtxt = f", weight {(weights[-1].data or {}).get('grams')} g" if weights else ""
        label = target_date.strftime("%b %d") if target_date else "recent"
        return {"answer": f"{label}: {len(feeds)} feeds ({ml} ml), {wet} wet, {dirty} dirty{wtxt}."}

    # fallback
    recent = events[-5:]
    lines = [f"{e.local.strftime('%b %d %H:%M')} — {e.type}" for e in recent]
    return {"answer": "Recent: " + "; ".join(lines) if lines else "No events yet."}

class AgentAskIn(BaseModel):
    question: str
    history: List[dict] = []
    tz: str = None


@app.post("/api/agent/ask")
async def agent_ask(body: AgentAskIn):
    """Run the Ask agent and stream each step as Server-Sent Events."""
    async def stream():
        db = SessionLocal()
        try:
            async for event in agent.run(db, body.question.strip(), body.history, body.tz):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'Agent crashed: {e}'})}\n\n"
        finally:
            db.close()
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/rag/query")
async def query_rag(q: str = "", db: Session = Depends(get_db)):
    """Answer a question from uploaded documents, with page citations."""
    if not q.strip():
        return {"answer": "Please enter a question.", "citations": [], "source": "none"}
    if not db.query(Chunk).first():
        return {"answer": "No documents uploaded yet. Go to Docs and upload a PDF.", "citations": [], "source": "none"}

    hits = rag.search(db, q, k=6)
    if not hits:
        return {"answer": NOT_FOUND_ANSWER, "citations": [], "source": "search"}

    citations = [{"n": i + 1, "title": title, "page": c.page, "snippet": c.content[:300]}
                 for i, (_, c, title) in enumerate(hits)]
    context = "\n\n".join(f"[{i + 1}] ({title}, page {c.page})\n{c.content}"
                          for i, (_, c, title) in enumerate(hits))
    try:
        message = await llm.chat(
            [{"role": "system", "content": RAG_SYSTEM_PROMPT},
             {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {q}"}],
            timeout=120.0, temperature=0.2, num_predict=400)
        answer = (message.get("content") or "").strip()
    except llm.LLMError as e:
        # LLM down: still show the best matching passages.
        return {"answer": "The local AI model is unavailable, so here are the closest passages from your documents.",
                "citations": citations[:3], "source": "fallback", "error": str(e)}

    if not answer or "NOT_FOUND" in answer:
        return {"answer": NOT_FOUND_ANSWER, "citations": [], "source": "llm"}
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited = [c for c in citations if c["n"] in used] or citations[:1]
    return {"answer": answer, "citations": cited, "source": "llm"}

@app.get("/api/handoff")
async def handoff(hours: int = 8, db: Session = Depends(get_db)):
    """Partner shift handoff briefing."""
    now = datetime.utcnow()
    shift_start = now - timedelta(hours=hours)
    baseline_start = now - timedelta(days=7)

    shift_events = db.query(Event).filter(Event.timestamp >= shift_start).all()
    baseline_events = db.query(Event).filter(Event.timestamp >= baseline_start, Event.timestamp < shift_start).all()

    def stats(evts):
        feeds = [e for e in evts if e.type == "feed"]
        milk = sum((e.data or {}).get("amount_ml", 0) for e in feeds)
        diapers = len([e for e in evts if e.type == "diaper"])
        sleep_min = sum((e.data or {}).get("duration_sec", 0) for e in evts if e.type == "sleep") / 60
        return {"feeds": len(feeds), "milk_ml": milk, "diapers": diapers, "sleep_min": round(sleep_min)}

    shift = stats(shift_events)
    base_raw = stats(baseline_events)
    # daily avg / 3 for comparable window
    base = {k: round(v / 7 / 3, 1) if k != "feeds" else round(v / 7 / 3) for k, v in base_raw.items()}

    deltas = {}
    for k in shift:
        b = base.get(k) or 1
        deltas[k] = round((shift[k] - b) / b * 100) if b else 0

    notes = [(e.data or {}).get("text") or (e.data or {}).get("note") for e in shift_events if e.type == "note"][:5]

    prompt = f"""Write a 3-sentence partner shift handoff.
Current shift ({hours}h): {shift}
Baseline (7-day avg scaled): {base}
Deltas %: {deltas}
Notes: {notes}
Be calm and practical. Mention what happened, how it compares, and what to watch."""

    briefing = "Shift summary unavailable."
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            })
            briefing = resp.json().get("response", briefing)
    except Exception:
        pass

    return {"shift": shift, "baseline": base, "deltas": deltas, "briefing": briefing}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "weight_g": {"type": ["number", "null"]},
        "length_cm": {"type": ["number", "null"]},
        "head_circumference_cm": {"type": ["number", "null"]},
        "feeding_plan_updates": {"type": "string"},
        "medications": {"type": "string"},
        "red_flags": {"type": "string"},
        "next_visit": {"type": "string"},
    },
    "required": ["summary", "weight_g", "length_cm", "head_circumference_cm",
                 "feeding_plan_updates", "medications", "red_flags", "next_visit"],
}

SCRIBE_SYSTEM_PROMPT = """You read the transcript of a baby's clinic or public health nurse visit and extract the care plan as JSON.
- summary: 2-3 plain sentences on what was discussed.
- weight_g: baby's weight in grams (convert kg x1000, lb x453.6), or null if not said.
- length_cm, head_circumference_cm: numbers in cm, or null if not said.
- feeding_plan_updates, medications, red_flags (warning signs to watch for), next_visit: short text, or "" if not mentioned.
Only use what is in the transcript. Never invent numbers."""

EMPTY_PLAN = {k: (None if k.endswith(("_g", "_cm")) else "") for k in PLAN_SCHEMA["properties"]}

# In-memory job table; fine for a single self-hosted process.
SCRIBE_JOBS: dict = {}
_background_tasks: set = set()


async def _run_scribe(job_id: str, content: bytes, filename: str):
    job = SCRIBE_JOBS[job_id]
    try:
        transcript = await audio.transcribe(content, filename)
    except Exception as e:
        job.update(status="error", error=f"Transcription failed: {e}")
        return
    if not transcript:
        job.update(status="error", error="No speech was detected in the recording.")
        return

    job.update(status="summarizing", transcript=transcript)
    plan, plan_error = dict(EMPTY_PLAN), None
    try:
        message = await llm.chat(
            [{"role": "system", "content": SCRIBE_SYSTEM_PROMPT},
             {"role": "user", "content": f"Transcript:\n{transcript[:24000]}"}],
            format=PLAN_SCHEMA, timeout=300.0)
        parsed = json.loads(message.get("content") or "{}")
        plan.update({k: parsed.get(k, plan[k]) for k in plan})
        for k in ("weight_g", "length_cm", "head_circumference_cm"):
            plan[k] = to_number(plan[k], None)
    except (llm.LLMError, json.JSONDecodeError) as e:
        plan_error = f"Could not extract a plan: {e}"

    db = SessionLocal()
    try:
        visit = Event(type="visit", data={"transcript": transcript, "plan": plan}, source="scribe")
        db.add(visit)
        db.commit()
        visit_id = visit.id
    finally:
        db.close()
    job.update(status="done", plan=plan, plan_error=plan_error, event_id=visit_id)


@app.post("/api/visits/scribe")
async def clinic_scribe(file: UploadFile = File(...)):
    """Start transcribing a clinic visit recording. Poll /api/visits/scribe/{job_id} for the result."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    job_id = uuid.uuid4().hex
    SCRIBE_JOBS[job_id] = {"status": "transcribing", "created": datetime.utcnow().isoformat()}
    task = asyncio.create_task(_run_scribe(job_id, content, file.filename or "visit.webm"))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id, "status": "transcribing"}


@app.get("/api/visits/scribe/{job_id}")
def clinic_scribe_status(job_id: str):
    job = SCRIBE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job (the server may have restarted).")
    return job

@app.get("/api/visit-brief")
def visit_brief(tz: str = None, db: Session = Depends(get_db)):
    """Health line / clinic visit brief — last 72h totals + flags + notes."""
    zone = get_tz(tz)
    profile = baby.get_profile(db)
    shots = baby.list_immunizations(db)
    now = datetime.utcnow()
    h72 = now - timedelta(hours=72)
    h48 = now - timedelta(hours=48)
    h24 = now - timedelta(hours=24)

    events = db.query(Event).filter(Event.timestamp >= h72).order_by(Event.timestamp).all()

    def count_in(window_start, etype, subtype=None):
        filtered = [e for e in events if e.timestamp >= window_start and e.type == etype]
        if subtype:
            filtered = [e for e in filtered if (e.data or {}).get("type") == subtype]
        return len(filtered)

    feeds_24 = count_in(h24, "feed")
    feeds_48 = count_in(h48, "feed")
    feeds_72 = count_in(h72, "feed")
    wet_24 = count_in(h24, "diaper", "wet")
    wet_48 = count_in(h48, "diaper", "wet")
    wet_72 = count_in(h72, "diaper", "wet")
    dirty_24 = count_in(h24, "diaper", "dirty")
    dirty_48 = count_in(h48, "diaper", "dirty")
    dirty_72 = count_in(h72, "diaper", "dirty")

    # Average feed interval (last 24h)
    feed_times = sorted([e.timestamp for e in events if e.type == "feed" and e.timestamp >= h24])
    avg_interval = None
    if len(feed_times) > 1:
        gaps = [(feed_times[i+1] - feed_times[i]).total_seconds() / 3600 for i in range(len(feed_times)-1)]
        avg_interval = round(sum(gaps) / len(gaps), 1)

    # Sleep
    sleep_ends = [e for e in events if e.type == "sleep" and (e.data or {}).get("action") == "end" and e.timestamp >= h24]
    total_sleep_min = sum((e.data or {}).get("duration_sec", 0) for e in sleep_ends) / 60
    longest_sleep = max((e.data or {}).get("duration_sec", 0) for e in sleep_ends) / 60 if sleep_ends else 0

    # Flags
    flags = []
    if wet_24 < 6:
        flags.append(f"{wet_24} wet diapers in last 24h — below the typical 6+ after the first few days.")
    if feeds_24 and avg_interval and avg_interval > 4:
        flags.append(f"Average feed interval ~{avg_interval}h in last 24h.")

    # Notes
    notes = [e for e in events if e.type == "note" and e.timestamp >= h48]
    note_texts = [(e.data or {}).get("text", "") for e in notes if (e.data or {}).get("text")]

    # Milk totals
    def milk_ml(window_start, milk_type):
        return sum((e.data or {}).get("amount_ml", 0) for e in events
                   if e.timestamp >= window_start and e.type == "feed" and (e.data or {}).get("type") == milk_type)

    breast_24 = milk_ml(h24, "breast")
    breast_48 = milk_ml(h48, "breast")
    breast_72 = milk_ml(h72, "breast")
    formula_24 = milk_ml(h24, "formula")
    formula_48 = milk_ml(h48, "formula")
    formula_72 = milk_ml(h72, "formula")

    # Formula names used
    formula_names = list({(e.data or {}).get("formula_name") for e in events if e.type == "feed" and (e.data or {}).get("formula_name")})

    # Detailed lists for expand
    def list_in(window_start, etype):
        return [{"time": utc_iso(e.timestamp), "data": e.data} for e in events if e.timestamp >= window_start and e.type == etype]

    brief = {
        "generated": utc_iso(now),
        "totals": {
            "feeds": {"24h": feeds_24, "48h": feeds_48, "72h": feeds_72, "avg_interval_h": avg_interval},
            "breast_ml": {"24h": breast_24, "48h": breast_48, "72h": breast_72},
            "formula_ml": {"24h": formula_24, "48h": formula_48, "72h": formula_72},
            "formula_names": formula_names,
            "wet": {"24h": wet_24, "48h": wet_48, "72h": wet_72},
            "dirty": {"24h": dirty_24, "48h": dirty_48, "72h": dirty_72},
            "sleep_24h_min": round(total_sleep_min),
            "longest_sleep_min": round(longest_sleep)
        },
        "flags": flags,
        "notes": note_texts[:10],
        "details": {
            "feeds": list_in(h72, "feed"),
            "diapers": list_in(h72, "diaper"),
            "weights": list_in(h72, "weight")
        },
        "copy_text": "",
        "profile": profile,
        "immunizations": shots,
    }

    lines = [
        "Baby Companion Visit Brief",
        f"Generated: {to_local(now, zone).strftime('%Y-%m-%d %H:%M %Z')} ({zone.key})",
        "",
    ]
    if profile:
        born = profile.get("birth_date")
        lines.append(" · ".join(x for x in [
            profile.get("name", ""),
            f"born {born} ({baby.age_text(born, zone)})" if born else "",
            f"birth weight {profile['birth_weight_g']} g" if profile.get("birth_weight_g") else "",
        ] if x))
    if shots:
        lines.append("Immunizations: " + "; ".join(f"{i['name']} ({i['date']})" for i in shots))
    if profile or shots:
        lines.append("")
    lines += [
        f"Feeds — 24h: {feeds_24} | 48h: {feeds_48} | 72h: {feeds_72}",
        f"Mom’s Milk — 24h: {breast_24} ml | 48h: {breast_48} ml | 72h: {breast_72} ml",
        f"Formula Milk — 24h: {formula_24} ml | 48h: {formula_48} ml | 72h: {formula_72} ml",
        f"Formula names: {', '.join(formula_names) if formula_names else 'n/a'}",
        f"Avg feed interval (24h): {avg_interval or 'n/a'} hours",
        f"Wet diapers — 24h: {wet_24} | 48h: {wet_48} | 72h: {wet_72}",
        f"Dirty diapers — 24h: {dirty_24} | 48h: {dirty_48} | 72h: {dirty_72}",
        f"Sleep (24h): {round(total_sleep_min)} min total, longest stretch {round(longest_sleep)} min",
        ""
    ]
    if flags:
        lines.append("Flags:")
        lines.extend(f"- {f}" for f in flags)
        lines.append("")
    if note_texts:
        lines.append("Recent notes:")
        lines.extend(f"- {n}" for n in note_texts[:5])
    brief["copy_text"] = "\n".join(lines)
    return brief

# ---------------------------------------------------------------- baby profile & immunizations

@app.get("/api/profile")
def read_profile(tz: str = None, db: Session = Depends(get_db)):
    p = baby.get_profile(db)
    return {"profile": p, "age": baby.age_text(p.get("birth_date"), get_tz(tz)) if p.get("birth_date") else ""}


@app.put("/api/profile")
def update_profile(data: dict, tz: str = None, db: Session = Depends(get_db)):
    try:
        p = baby.save_profile(db, data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Birth date must be a valid date.")
    return {"profile": p, "age": baby.age_text(p.get("birth_date"), get_tz(tz)) if p.get("birth_date") else ""}


class ImmunizationIn(BaseModel):
    name: str
    date: str
    place: str = ""
    notes: str = ""
    next_due: str = ""
    tz: str = None


@app.get("/api/immunizations")
def get_immunizations(db: Session = Depends(get_db)):
    return baby.list_immunizations(db)


@app.post("/api/immunizations")
def create_immunization(body: ImmunizationIn, db: Session = Depends(get_db)):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Enter the vaccine name.")
    try:
        e = baby.add_immunization(db, body.name, body.date, get_tz(body.tz), body.place, body.notes, body.next_due)
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be valid (YYYY-MM-DD).")
    return {"id": e.id, **e.data}


@app.delete("/api/immunizations/{event_id}")
def delete_immunization(event_id: int, db: Session = Depends(get_db)):
    e = db.get(Event, event_id)
    if not e or e.type != "immunization":
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(e)
    db.commit()
    return {"deleted": event_id}


@app.get("/insights")
def insights(db: Session = Depends(get_db)):
    """Simple pattern flags from recent logs."""
    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)
    events = db.query(Event).filter(Event.timestamp >= day_ago).all()

    feeds = [e for e in events if e.type == "feed"]
    diapers = [e for e in events if e.type == "diaper"]
    wet = [e for e in diapers if (e.data or {}).get("type") == "wet"]

    flags = []
    if len(wet) < 6:
        flags.append("Fewer than 6 wet diapers in the last 24h. Public health guidance often looks for 6+ after the first week.")
    if feeds:
        last_feed = max(feeds, key=lambda e: e.timestamp)
        gap = (now - last_feed.timestamp).total_seconds() / 3600
        if gap > 4:
            flags.append(f"About {gap:.1f} hours since last feed. Consider offering a feed if baby is showing hunger cues.")
    if not feeds and not diapers:
        flags.append("No events logged in the last 24h.")

    latest_weight = None
    weights = db.query(Event).filter(Event.type == "weight").order_by(Event.timestamp.desc()).first()
    if weights:
        latest_weight = (weights.data or {}).get("grams")

    return {"flags": flags, "feeds_24h": len(feeds), "wet_24h": len(wet), "latest_weight": latest_weight}

@app.get("/summary/today")
def summary_today(tz: str = None, db: Session = Depends(get_db)):
    # Midnight in the user's time zone, converted to UTC for the query.
    local_midnight = datetime.now(get_tz(tz)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = local_midnight.astimezone(timezone.utc).replace(tzinfo=None)
    events = db.query(Event).filter(Event.timestamp >= start).order_by(Event.timestamp).all()

    feeds = sum(1 for e in events if e.type == "feed")
    wet = sum(1 for e in events if e.type == "diaper" and (e.data or {}).get("type") == "wet")
    dirty = sum(1 for e in events if e.type == "diaper" and (e.data or {}).get("type") == "dirty")
    sleep_starts = sum(1 for e in events if e.type == "sleep" and (e.data or {}).get("action") == "start")

    breast_ml = sum((e.data or {}).get("amount_ml", 0) for e in events if e.type == "feed" and (e.data or {}).get("type") == "breast")
    formula_ml = sum((e.data or {}).get("amount_ml", 0) for e in events if e.type == "feed" and (e.data or {}).get("type") == "formula")
    formula_feeds = [e for e in events if e.type == "feed" and (e.data or {}).get("type") == "formula"]
    formula_names = list({(e.data or {}).get("formula_name") for e in formula_feeds if (e.data or {}).get("formula_name")})
    weights = [e for e in events if e.type == "weight"]
    latest_w = (weights[-1].data or {}).get("grams") if weights else None
    text = f"Today: {feeds} feeds, Mom’s Milk {breast_ml} ml, Formula {formula_ml} ml, {wet} wet, {dirty} dirty, {sleep_starts} sleep sessions."
    if latest_w:
        text += f" Latest weight: {latest_w} g."
    if formula_names:
        text += f" Formula names: {', '.join(formula_names)}."
    if not events:
        text = "No events logged today yet."
    return {"summary": text, "events": len(events)}

@app.get("/mcp/tools")
def mcp_tools():
    return {
        "tools": [
            {"name": "log_event", "description": "Log a baby event", "endpoint": "/events"},
            {"name": "query_log", "description": "Query recent events", "endpoint": "/events"},
            {"name": "query_docs", "description": "Query local health documents", "endpoint": "/rag/query"},
            {"name": "get_insights", "description": "Get pattern flags", "endpoint": "/insights"},
            {"name": "daily_summary", "description": "Get today's summary", "endpoint": "/summary/today"}
        ]
    }

@app.post("/mcp/call")
def mcp_call(name: str, payload: dict = None, db: Session = Depends(get_db)):
    """Simple MCP-style call endpoint."""
    if name == "log_event" and payload:
        return create_event(EventCreate(**payload), BackgroundTasks(), db)
    if name == "query_log":
        return list_events(50, db)
    if name == "get_insights":
        return insights(db)
    if name == "daily_summary":
        return summary_today(db)
    return {"error": "unknown tool"}

# Mounted last so it doesn't shadow the /mcp/tools and /mcp/call routes above.
app.mount("/mcp", mcp_server.sse_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

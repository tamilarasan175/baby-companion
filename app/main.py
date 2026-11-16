from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import os
import re
import math
import httpx
import fitz  # PyMuPDF
from io import BytesIO
from .models import SessionLocal, Event, Document, Chunk
from .services.care_tools import log_feed, log_diaper, set_reminder, get_recent_context
from .services.mcp_server import mcp as mcp_server

app = FastAPI(title="Baby Companion")
app.mount("/mcp", mcp_server.sse_app())
app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

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
    class Config:
        from_attributes = True

def simple_embed(text: str) -> list:
    """Lightweight bag-of-words embedding for semantic-ish search without extra deps."""
    words = re.findall(r'\w+', text.lower())
    vec = {}
    for w in words:
        if len(w) > 2:
            vec[w] = vec.get(w, 0) + 1
    # Normalize
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1
    return {k: v/norm for k, v in vec.items()}

def cosine(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    return sum(a[k] * b[k] for k in keys)

def chunk_text(text: str, size: int = 600, overlap: int = 100) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/api/voice")
async def voice_transcribe(file: UploadFile = File(...)):
    """Transcribe short voice note and return text."""
    import tempfile
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    transcript = ""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        # convert to wav to avoid codec issues
        wav_path = tmp_path + ".wav"
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path], capture_output=True)
        segments, _ = model.transcribe(wav_path if os.path.exists(wav_path) else tmp_path)
        transcript = " ".join(s.text for s in segments).strip()
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception as e:
        transcript = ""
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
    return {"transcript": transcript}

@app.post("/api/voice-agent")
async def voice_agent(transcript: str, db: Session = Depends(get_db)):
    """Multi-intent voice router. Sends transcript to Ollama and executes tool calls."""
    tools = [
        {"type": "function", "function": {"name": "log_feed", "description": "Log a milk feed. milk_type must be 'breast' or 'formula'. Put brand names like Enfamil in formula_name.", "parameters": {"type": "object", "properties": {"milk_type": {"type": "string", "enum": ["breast", "formula"]}, "ml": {"type": "integer"}, "formula_name": {"type": "string"}, "duration_min": {"type": "number"}}, "required": ["milk_type"]}}},
        {"type": "function", "function": {"name": "log_diaper", "description": "Log a diaper", "parameters": {"type": "object", "properties": {"status": {"type": "string"}, "note": {"type": "string"}}, "required": ["status"]}}},
        {"type": "function", "function": {"name": "set_reminder", "description": "Set a reminder", "parameters": {"type": "object", "properties": {"label": {"type": "string"}, "delay_minutes": {"type": "integer"}}, "required": ["label", "delay_minutes"]}}}
    ]
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": transcript}],
                "tools": tools,
                "stream": False
            })
            data = resp.json()
            tool_calls = data.get("message", {}).get("tool_calls", [])
            results = []
            for tc in tool_calls:
                name = tc.get("function", {}).get("name")
                args = tc.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                if name == "log_feed":
                    results.append(log_feed(db, milk_type=args.get("milk_type", "breast"), ml=args.get("ml", 0), formula_name=args.get("formula_name"), duration_min=args.get("duration_min")))
                elif name == "log_diaper":
                    results.append(log_diaper(db, status=args.get("status", "wet"), note=args.get("note")))
                elif name == "set_reminder":
                    results.append(set_reminder(label=args.get("label", "check"), delay_minutes=args.get("delay_minutes", 120)))
            summary = " · ".join([str(r.get("logged") or r.get("reminder") or r) for r in results]) or "No actions"
            return {"summary": summary, "results": results}
    except Exception as e:
        return {"summary": f"Agent error: {str(e)}", "results": []}

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
    os.makedirs("docs", exist_ok=True)
    path = f"docs/{file.filename}"
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    text = ""
    page_texts = []
    if file.filename.lower().endswith(".pdf"):
        try:
            doc_pdf = fitz.open(stream=content, filetype="pdf")
            for i, page in enumerate(doc_pdf):
                # sort=True keeps reading order; blocks mode helps with columns
                page_text = page.get_text("text", sort=True)
                page_texts.append((i + 1, page_text))
                text += page_text + "\n"
            doc_pdf.close()
        except Exception:
            text = "[PDF text extraction failed]"
            page_texts = [(1, text)]
    elif file.filename.lower().endswith((".txt", ".md")):
        text = content.decode("utf-8", errors="ignore")
        page_texts = [(1, text)]
    else:
        text = "[Unsupported file type]"
        page_texts = [(1, text)]

    doc = Document(title=file.filename, source_path=path, content=text)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Chunk by page so we keep page numbers
    for page_num, page_text in page_texts:
        for chunk in chunk_text(page_text):
            emb = simple_embed(chunk)
            db.add(Chunk(document_id=doc.id, content=chunk, page=page_num, embedding=emb))
    db.commit()
    return {"id": doc.id, "title": doc.title, "chars": len(text), "chunks": db.query(Chunk).filter(Chunk.document_id == doc.id).count()}

@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()

async def ask_ollama(question: str, context: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"""You are ABC, an Alberta baby companion. Answer the question using ONLY the information in the excerpt below.
- Start directly with the answer. Do not use conversational filler.
- Only include information directly related to the user's specific question. Ignore unrelated sidebars.
- Pay close attention to section headings inside tables. Do not mix up opposite columns (for example, do not list 'signs baby is full' under 'hunger cues').
- If the user asks a multi-part question and the text only answers one part, answer only the supported part and state that the other part was not found.
- Use clear bullet points.
- If the retrieved chunks do not contain the answer, respond ONLY with: "I could not find this in the uploaded Alberta Health documents. Please call Health Link at 811 or check with your healthcare provider." Do NOT include a Source line.
- If you do find the answer, end with exactly: Source: The Early Years, p. X

Excerpt:
{context}

Question: {question}

Answer:""",
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 300}
                }
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
    except Exception:
        return None
    return None

def split_query(q: str) -> list:
    """Split multi-part questions on ? or and/or."""
    parts = re.split(r'\?|\band\b|\bor\b', q, flags=re.IGNORECASE)
    return [p.strip() for p in parts if len(p.strip()) > 10]

def score_chunks(subq: str, chunks) -> list:
    q_lower = subq.lower()
    words = [w for w in re.split(r'\W+', q_lower) if len(w) > 2]
    scored = []
    for c in chunks:
        if not c.content:
            continue
        text = c.content.lower()
        match_count = sum(1 for w in words if w in text)
        score = match_count * 3
        phrases = [
            "hunger cues", "rooting", "vitamin d", "400 iu",
            "diaper rash", "wet diapers", "dirty diapers",
            "safe sleep", "fever", "temperature",
            "vegetable", "fruit", "food guide", "serving"
        ]
        for phrase in phrases:
            if phrase in q_lower and phrase in text:
                score += 12
        # always include, even with zero matches, so something is returned
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:6]

@app.get("/api/logs/ask")
def ask_logs(q: str, db: Session = Depends(get_db)):
    """Answer questions about past logs, including dates."""
    q_lower = q.lower()
    events = db.query(Event).order_by(Event.timestamp).all()

    # Try to extract a date like "nov 16", "november 8", "11/16"
    import re
    from datetime import datetime
    target_date = None
    m = re.search(r'(nov|november|oct|october|dec|december|jan|january)\s+(\d{1,2})', q_lower)
    if m:
        month_map = {"jan":1,"january":1,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}
        mon = month_map.get(m.group(1), 11)
        day = int(m.group(2))
        year = datetime.utcnow().year
        try:
            target_date = datetime(year, mon, day)
        except:
            pass

    def on_day(e, d):
        if not d:
            return True
        return e.timestamp.date() == d.date()

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
        return {"answer": f"Weight on {w.timestamp.strftime('%b %d')}: {(w.data or {}).get('grams','?')} g"}

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
    lines = [f"{e.timestamp.strftime('%b %d %H:%M')} — {e.type}" for e in recent]
    return {"answer": "Recent: " + "; ".join(lines) if lines else "No events yet."}
async def query_rag(q: str, db: Session = Depends(get_db)):
    if not q or not q.strip():
        return {"answer": "Please enter a question.", "points": [], "citations": []}

    chunks = db.query(Chunk).all()
    if not chunks:
        return {"answer": "No documents uploaded yet. Go to Docs and upload a PDF.", "points": [], "citations": []}
    subqueries = split_query(q)
    if not subqueries:
        subqueries = [q]

    # Collect top chunks from each sub-query, deduplicate
    seen = set()
    all_top = []
    for subq in subqueries:
        for score, c in score_chunks(subq, chunks):
            if c.id not in seen:
                seen.add(c.id)
                all_top.append((score, c))

    all_top.sort(key=lambda x: x[0], reverse=True)
    top = all_top[:6]

    if not top:
        return {"answer": f"No relevant section found for “{q}”.", "points": [], "citations": []}

    context_parts = []
    for _, c in top:
        page_info = f"(page {c.page})" if c.page else ""
        context_parts.append(f"{page_info}\n{c.content}")
    context = "\n\n---\n\n".join(context_parts)
    llm = await ask_ollama(q, context)

    if llm and len(llm.strip()) > 15:
        lines = [line.strip(" -•*") for line in llm.split("\n") if line.strip()]
        points = [l for l in lines if len(l) > 5]
        if "could not find" in llm.lower() or "not found" in llm.lower():
            return {
                "answer": llm.strip(),
                "points": [],
                "citations": [],
                "source": "ollama"
            }
        page = top[0][1].page
        return {
            "answer": "Grounded answer:",
            "points": points[:12] or [llm],
            "citations": [{"title": "document", "snippet": top[0][1].content[:400], "page": page}],
            "source": "ollama"
        }

    best = top[0][1].content
    sentences = re.split(r'(?<=[.!?])\s+', best)
    points = [re.sub(r'\s+', ' ', s).strip() for s in sentences if len(s.strip()) > 40][:5]
    return {
        "answer": "From documents (LLM unavailable):",
        "points": points or [re.sub(r'\s+', ' ', best)[:500]],
        "citations": [{"title": "document", "snippet": best[:400], "page": top[0][1].page}],
        "source": "fallback"
    }

from faster_whisper import WhisperModel

# Load a small model once
whisper_model = None
def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return whisper_model

@app.post("/voice")
async def voice_transcribe(file: UploadFile = File(...)):
    """Accept audio, transcribe with Whisper, return text."""
    audio_bytes = await file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        model = get_whisper()
        segments, _ = model.transcribe(tmp_path, language="en")
        text = " ".join(s.text for s in segments).strip().lower()
    except Exception as e:
        text = ""
    finally:
        os.unlink(tmp_path)
    return {"text": text}

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

@app.post("/api/visits/scribe")
async def clinic_scribe(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Transcribe clinic visit audio and extract care plan."""
    import tempfile
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        wav_path = tmp_path + ".wav"
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path], capture_output=True)
        segments, _ = model.transcribe(wav_path if os.path.exists(wav_path) else tmp_path)
        transcript = " ".join(s.text for s in segments)
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception as e:
        transcript = f"[transcription failed: {e}]"
    finally:
        os.remove(tmp_path)

    plan = {"weight_g": None, "length_cm": None, "feeding_plan_updates": "", "medications": "", "red_flags": "", "next_visit": ""}
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": f"Extract a care plan as JSON only: {{weight_g, length_cm, feeding_plan_updates, medications, red_flags, next_visit}}. Transcript: {transcript[:4000]}",
                "stream": False,
                "options": {"temperature": 0.1}
            })
            text = resp.json().get("response", "")
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                import json
                plan = json.loads(text[start:end])
    except Exception:
        pass

    return {"transcript": transcript[:2000], "plan": plan}

@app.get("/api/visit-brief")
def visit_brief(db: Session = Depends(get_db)):
    """811 / Clinic Visit Brief — last 72h totals + flags + notes."""
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
        flags.append(f"{wet_24} wet diapers in last 24h — below the typical 6+ AHS benchmark after the first few days.")
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
        return [{"time": e.timestamp.isoformat(), "data": e.data} for e in events if e.timestamp >= window_start and e.type == etype]

    brief = {
        "generated": now.isoformat(),
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
        "copy_text": ""
    }

    lines = [
        "ABC Visit Brief (MST)",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M')} UTC",
        "",
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

@app.get("/insights")
def insights(db: Session = Depends(get_db)):
    """Simple Alberta-grounded pattern flags from recent logs."""
    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)
    events = db.query(Event).filter(Event.timestamp >= day_ago).all()

    feeds = [e for e in events if e.type == "feed"]
    diapers = [e for e in events if e.type == "diaper"]
    wet = [e for e in diapers if (e.data or {}).get("type") == "wet"]

    flags = []
    if len(wet) < 6:
        flags.append("Fewer than 6 wet diapers in the last 24h. Alberta guidance often looks for 6+ after the first few days.")
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
def summary_today(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
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
        return create_event(EventCreate(**payload), db)
    if name == "query_log":
        return list_events(50, db)
    if name == "get_insights":
        return insights(db)
    if name == "daily_summary":
        return summary_today(db)
    return {"error": "unknown tool"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

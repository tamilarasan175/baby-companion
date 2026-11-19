"""Voice log agent: turns a spoken sentence into one or more log entries.

1. Ask the local LLM which events were mentioned (tool calls, one per event).
2. Ground every call against the transcript: amounts, units and brand names are
   parsed by code, and a call is dropped if the words it relies on were never said.
   Small local models (llama3.2 3B) otherwise invent amounts, brands and diapers.
3. If the LLM is unavailable or yields nothing usable, use the rule-based parser.
4. If nothing matches at all, save the transcript as a note so nothing is lost.
"""
import json
import re

from sqlalchemy.orm import Session

from . import llm
from .care_tools import log_diaper, log_feed, log_note, log_sleep, log_weight, set_reminder

# Helps Whisper spell domain words (e.g. "Enfamil" instead of "infamil").
WHISPER_HINT = "Baby log: breast milk, formula, Enfamil, Similac, Kendamil, ml, ounces, wet diaper, dirty diaper, poop, nap."

SYSTEM_PROMPT = """You record events from a parent's spoken baby-care note by calling tools.
Call one tool per event that was actually said. Do not add events that were not mentioned.
Do not guess amounts or brand names. If something does not fit a tool, use log_note."""

TOOLS = [
    {"type": "function", "function": {
        "name": "log_feed", "description": "A milk feed: breast milk (nursing, pumped) or formula.",
        "parameters": {"type": "object", "properties": {
            "milk_type": {"type": "string", "enum": ["breast", "formula"]},
            "duration_min": {"type": "number", "description": "Feed length in minutes, only if said"}},
            "required": ["milk_type"]}}},
    {"type": "function", "function": {
        "name": "log_diaper", "description": "A diaper change. wet = pee, dirty = poop.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["wet", "dirty"]}},
            "required": ["status"]}}},
    {"type": "function", "function": {
        "name": "log_sleep", "description": "Baby fell asleep (start) or woke up (end).",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["start", "end"]}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "log_weight", "description": "Baby's weight was measured.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "log_note", "description": "Anything else: symptoms, mood, medicine, questions.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "set_reminder", "description": "Remind the parent to do something later.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string"},
            "delay_minutes": {"type": "integer"}},
            "required": ["label", "delay_minutes"]}}},
]

WET_RE = re.compile(r"\b(wet|pee|peed|pees|urine|urinated)\b")
DIRTY_RE = re.compile(r"\b(dirty|poop|pooped|poopy|poo|stool|bm|bowel|soiled)\b")
DIAPER_RE = re.compile(r"\b(diaper|nappy)")
SLEEP_START_RE = re.compile(r"fell asleep|went (down|to sleep|to bed)|(nap|sleep)\w* (started|start)|is (sleeping|asleep|napping)|down for a nap|falling asleep")
SLEEP_END_RE = re.compile(r"\bwoke\b|woken|waking|is awake|got up|(nap|sleep) (ended|end|over|finished)")
FORMULA_RE = re.compile(r"formula|enfamil|infamil|similac|good start|kendamil|nutramigen|alimentum")
BREAST_RE = re.compile(r"breast|nurs(e|ed|ing)|mom'?s milk|mother'?s milk|pumped|expressed|latch")
FEED_RE = re.compile(r"\b(fed|feed|feeding|ate|eaten|drank|bottle|milk)\b")
BRANDS = {"enfamil": "Enfamil", "infamil": "Enfamil", "similac": "Similac", "kendamil": "Kendamil",
          "good start": "Good Start", "nutramigen": "Nutramigen", "alimentum": "Alimentum"}


def parse_ml(t: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ml|mls|milliliters?|millilitres?|cc)\b", t)
    if m:
        return round(float(m.group(1)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:oz|ounces?)\b", t)
    if m:
        return round(float(m.group(1)) * 30)
    return None


def parse_grams(t: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|kilos?|grams?|g)\b", t)
    if m:
        val = float(m.group(1))
        return round(val * 1000 if m.group(2).startswith("k") else val)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)\b(?:\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*(?:oz|ounces?))?", t)
    if m:
        return round(float(m.group(1)) * 453.6 + float(m.group(3) or 0) * 28.35)
    return None


def parse_minutes(t: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(min|mins|minutes?|hours?|hrs?)\b", t)
    if not m:
        return None
    val = float(m.group(1))
    return val * 60 if m.group(2).startswith("h") else val


def parse_brand(t: str):
    for key, name in BRANDS.items():
        if key in t:
            return name
    return None


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                 "sixty": 60, "ninety": 90, "half": 0.5}


def normalize(transcript: str) -> str:
    t = transcript.lower()
    return re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: str(_NUMBER_WORDS[m.group(1)]), t)


def ground(name: str, args: dict, t: str):
    """Check an LLM tool call against the transcript. Returns fixed-up args, or None to drop it."""
    if name == "log_feed":
        if not (FORMULA_RE.search(t) or BREAST_RE.search(t) or FEED_RE.search(t) or parse_ml(t)):
            return None
        milk = args.get("milk_type")
        if milk == "formula" and not FORMULA_RE.search(t):
            milk = "breast"
        elif milk != "formula" and FORMULA_RE.search(t) and not BREAST_RE.search(t):
            milk = "formula"
        return {"milk_type": milk or "breast", "ml": parse_ml(t), "duration_min": parse_minutes(t),
                "formula_name": parse_brand(t) if milk == "formula" else None}
    if name == "log_diaper":
        status = args.get("status")
        if status == "dirty":
            return args if DIRTY_RE.search(t) else None
        # A plain "changed a diaper" counts as wet.
        return {"status": "wet"} if WET_RE.search(t) or (DIAPER_RE.search(t) and not DIRTY_RE.search(t)) else None
    if name == "log_sleep":
        if args.get("action") == "end":
            return args if SLEEP_END_RE.search(t) else None
        return {"action": "start"} if SLEEP_START_RE.search(t) else None
    if name == "log_weight":
        grams = parse_grams(t)
        return {"grams": grams} if grams and "weigh" in t else None
    return args


def rule_based_actions(t: str) -> list:
    """Keyword parser used when the LLM is unavailable. Returns [(tool_name, args)]."""
    actions = []
    if parse_ml(t) is not None or FORMULA_RE.search(t) or BREAST_RE.search(t) or FEED_RE.search(t):
        actions.append(("log_feed", {"milk_type": "formula" if FORMULA_RE.search(t) and not BREAST_RE.search(t) else "breast"}))
    if WET_RE.search(t) or (DIAPER_RE.search(t) and not DIRTY_RE.search(t)):
        actions.append(("log_diaper", {"status": "wet"}))
    if DIRTY_RE.search(t):
        actions.append(("log_diaper", {"status": "dirty"}))
    if SLEEP_START_RE.search(t):
        actions.append(("log_sleep", {"action": "start"}))
    elif SLEEP_END_RE.search(t):
        actions.append(("log_sleep", {"action": "end"}))
    if "weigh" in t:
        actions.append(("log_weight", {}))
    return actions


def _dispatch(db: Session, name: str, args: dict, transcript: str):
    if name == "log_feed":
        return log_feed(db, milk_type=args["milk_type"], ml=args.get("ml"),
                        formula_name=args.get("formula_name"), duration_min=args.get("duration_min"))
    if name == "log_diaper":
        return log_diaper(db, status=args["status"])
    if name == "log_sleep":
        return log_sleep(db, action=args["action"])
    if name == "log_weight":
        return log_weight(db, grams=args["grams"])
    if name == "log_note":
        return log_note(db, text=args.get("text") or transcript)
    if name == "set_reminder":
        return set_reminder(label=args.get("label", "check"), delay_minutes=args.get("delay_minutes", 120))
    return None


async def run(db: Session, transcript: str) -> dict:
    transcript = (transcript or "").strip()
    if not transcript:
        return {"summary": "Nothing heard", "results": [], "mode": "none"}
    t = normalize(transcript)

    raw_calls, llm_error = [], None
    try:
        message = await llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": transcript}],
            tools=TOOLS, timeout=60.0)
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            raw_calls.append((fn.get("name"), args))
    except llm.LLMError as e:
        llm_error = str(e)

    grounded, seen = [], set()

    def add(name, args):
        fixed = ground(name, dict(args), t)
        key = (name, json.dumps(fixed, sort_keys=True))
        if fixed is not None and key not in seen:
            seen.add(key)
            grounded.append((name, fixed))

    for name, args in raw_calls:
        add(name, args)
    from_llm = len(grounded)

    # The keyword parser also runs, to add events the model missed (e.g. the wet half of
    # "wet and dirty"). It only adds kinds of event the model didn't already log.
    def covered(name, args):
        for n, a in grounded:
            if n != name:
                continue
            if name == "log_diaper" and a.get("status") != args.get("status"):
                continue
            return True
        return False

    for name, args in rule_based_actions(t):
        if not covered(name, args):
            add(name, args)

    if from_llm and len(grounded) > from_llm:
        mode = "llm+rules"   # model did most of it; keywords filled a gap
    elif from_llm:
        mode = "llm"
    else:
        mode = "rules"       # model unreachable, or nothing it proposed matched what was said

    results = [r for r in (_dispatch(db, name, args, transcript) for name, args in grounded) if r]
    if not results:
        mode = "note"
        results = [log_note(db, transcript)]

    summary = " · ".join(r.get("text", "") for r in results)
    return {"summary": summary, "results": results, "mode": mode, "llm_error": llm_error}

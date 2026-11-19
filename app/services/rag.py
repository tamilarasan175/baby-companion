"""Document ingestion and retrieval for the Ask tab.

Retrieval uses BM25 keyword ranking (no extra dependencies). It replaces the
earlier substring counting, which matched "eat" inside "great", counted
filler words like "how" and "the", and always sent the top chunks to the LLM
even when none of them were relevant.
"""
import math
import os
import re
from collections import Counter

import pymupdf as fitz
from sqlalchemy.orm import Session

from ..models import Chunk, Document

SUPPORTED_EXTS = (".pdf", ".txt", ".md")

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours yourself yourselves
baby babies infant tell know please much many get also may might
""".split())


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def chunk_text(text: str, words_per_chunk: int = 140, overlap: int = 35) -> list:
    words = text.split()
    if not words:
        return []
    chunks, step = [], words_per_chunk - overlap
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + words_per_chunk])
        if len(piece) > 40:
            chunks.append(piece)
        if start + words_per_chunk >= len(words):
            break
    return chunks


def extract_pages(filename: str, content: bytes) -> list:
    """Return [(page_number, text)]."""
    name = filename.lower()
    if name.endswith(".pdf"):
        pages = []
        with fitz.open(stream=content, filetype="pdf") as pdf:
            for i, page in enumerate(pdf):
                pages.append((i + 1, page.get_text("text", sort=True)))
        return pages
    if name.endswith((".txt", ".md")):
        return [(1, content.decode("utf-8", errors="ignore"))]
    raise ValueError("Unsupported file type. Use PDF, TXT or MD.")


def ingest(db: Session, filename: str, content: bytes, source_path: str) -> Document:
    """Store a document and its chunks, replacing an earlier upload with the same name."""
    pages = extract_pages(filename, content)
    for old in db.query(Document).filter(Document.title == filename).all():
        db.query(Chunk).filter(Chunk.document_id == old.id).delete()
        db.delete(old)
    doc = Document(title=filename, source_path=source_path, content="\n".join(t for _, t in pages))
    db.add(doc)
    db.flush()
    for page_num, page_text in pages:
        for piece in chunk_text(page_text):
            db.add(Chunk(document_id=doc.id, content=piece, page=page_num))
    db.commit()
    db.refresh(doc)
    return doc


def ingest_folder(db: Session, folder: str = "docs") -> list:
    """Ingest files dropped into docs/ that are not in the database yet."""
    if not os.path.isdir(folder):
        return []
    known = {t for (t,) in db.query(Document.title).all()}
    added = []
    for name in sorted(os.listdir(folder)):
        if name in known or not name.lower().endswith(SUPPORTED_EXTS) or name.lower().startswith("readme"):
            continue  # README.txt is the folder's own instructions, not a health guide
        path = os.path.join(folder, name)
        try:
            with open(path, "rb") as f:
                ingest(db, name, f.read(), path)
            added.append(name)
        except Exception as e:
            db.rollback()
            print(f"[rag] could not ingest {name}: {e}")
    return added


def search(db: Session, question: str, k: int = 6) -> list:
    """Return up to k (score, chunk, doc_title) tuples ranked by BM25. Only chunks that match."""
    q_terms = list(dict.fromkeys(tokenize(question)))
    if not q_terms:
        return []
    rows = db.query(Chunk, Document.title).join(Document, Document.id == Chunk.document_id).all()
    corpus = [(c, title, tokenize(c.content or "")) for c, title in rows]
    if not corpus:
        return []

    n = len(corpus)
    avgdl = sum(len(toks) for _, _, toks in corpus) / n or 1
    df = Counter()
    for _, _, toks in corpus:
        df.update(set(toks))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
    q_bigrams = [f"{a} {b}" for a, b in zip(q_terms, q_terms[1:])]

    k1, b = 1.5, 0.75
    results = []
    for chunk, title, toks in corpus:
        tf = Counter(toks)
        score = 0.0
        for t in q_terms:
            if tf[t]:
                score += idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avgdl))
        if score <= 0:
            continue
        # Bonus when adjacent query words appear together, e.g. "hunger cues", "vitamin d".
        joined = " ".join(toks)
        score += sum(1.5 for bg in q_bigrams if bg in joined)
        results.append((score, chunk, title))
    results.sort(key=lambda r: r[0], reverse=True)
    if not results:
        return []
    # Drop weak tail matches (e.g. a single common word) relative to the best hit.
    best = results[0][0]
    return [r for r in results[:k] if r[0] >= best * 0.35]

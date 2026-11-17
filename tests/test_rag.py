from app.services import rag

GUIDE = (b"Hunger cues: Early hunger cues include rooting, sucking on hands and smacking lips. "
         b"Crying is a late hunger cue.\n\n" + b"filler text about other topics. " * 40 +
         b"\n\nWet diapers: After the first week, your baby should have at least 6 heavy wet diapers every 24 hours.")


def test_tokenize_drops_filler_and_stems():
    assert rag.tokenize("What are the hunger cues?") == ["hunger", "cue"]
    assert "diaper" in rag.tokenize("wet diapers")


def test_chunks_do_not_split_words():
    chunks = rag.chunk_text("word " * 500)
    assert len(chunks) > 1
    assert all(not c.startswith("ord") for c in chunks)


def test_search_finds_the_right_passage_and_skips_unrelated(db):
    rag.ingest(db, "guide.txt", GUIDE, "docs/guide.txt")
    hits = rag.search(db, "how many wet diapers per day?")
    assert hits and "wet diapers" in hits[0][1].content.lower()
    assert hits[0][2] == "guide.txt"
    assert rag.search(db, "which car seat should I buy?") == []


def test_reupload_replaces_instead_of_duplicating(db):
    rag.ingest(db, "guide.txt", GUIDE, "docs/guide.txt")
    rag.ingest(db, "guide.txt", GUIDE, "docs/guide.txt")
    from app.models import Document
    assert db.query(Document).filter(Document.title == "guide.txt").count() == 1

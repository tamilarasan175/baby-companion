"""Test setup: a throwaway SQLite database and an unreachable Ollama, set before the app is imported."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="baby-companion-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"  # nothing listens here: LLM calls fail fast
os.environ["DEFAULT_TZ"] = "America/Edmonton"
os.environ["DOCS_DIR"] = f"{_tmp}/docs"  # never index the real docs/ folder in tests

import pytest  # noqa: E402

from app.models import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

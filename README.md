# Baby Companion

Self-hosted, privacy-first baby logging and companion tool.

## Features (MVP)
- Structured logging: feeds, diapers, sleep, notes
- Dark mode mobile-friendly UI
- Voice input support (browser Web Speech API)
- Local document storage for RAG (citations)
- MCP endpoint for AI assistants
- Zero accounts, everything local

## Quick Start

```bash
docker compose up --build
```

Then open http://localhost:8000

## Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Data is stored in `data/baby.db`. Drop PDFs into `docs/` for future RAG.

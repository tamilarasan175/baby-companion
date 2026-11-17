"""Settings that differ by region or deployment. Override with environment variables."""
import os

# Region shown in prompts and messages (e.g. "Alberta", "Ontario", "UK").
REGION = os.getenv("REGION", "Alberta")
# Non-emergency nurse/health advice line, and the emergency number.
HEALTH_LINE = os.getenv("HEALTH_LINE", "Health Link 811")
HEALTH_LINE_NUMBER = os.getenv("HEALTH_LINE_NUMBER", "811")
EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "911")

# Optional password for the whole app (web UI, API and /mcp). Empty = no login (local use only).
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Folder with health documents for the Ask tab (indexed at startup).
DOCS_DIR = os.getenv("DOCS_DIR", "docs")

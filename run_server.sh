
#!/usr/bin/env bash
# Launches the QEM Lab web app (FastAPI backend + static frontend, one process/port).
set -e
cd "$(dirname "$0")"
echo "Starting QEM Lab at http://localhost:8000 ..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
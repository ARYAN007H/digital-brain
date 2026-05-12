#!/usr/bin/env bash
# ── Start all Digital Brain services ──────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/data"
mkdir -p "$PID_DIR"

source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
export OLLAMA_KEEP_ALIVE=0

echo "🧠 Starting Digital Brain..."

# ── PocketBase ────────────────────────────────────────────
if ! pgrep -f "pocketbase serve" > /dev/null; then
    echo "▸ Starting PocketBase..."
    "$PROJECT_ROOT/bin/pocketbase" serve \
        --dir "$PID_DIR/pocketbase" \
        > "$PID_DIR/pocketbase.log" 2>&1 &
    echo $! > "$PID_DIR/pocketbase.pid"
    echo "  PocketBase: http://localhost:8090"
else
    echo "  PocketBase already running"
fi

# ── Inbox Watcher ─────────────────────────────────────────
if ! pgrep -f "brain.ingestion.watcher" > /dev/null; then
    echo "▸ Starting inbox watcher..."
    cd "$PROJECT_ROOT"
    python -m brain.ingestion.watcher \
        > "$PID_DIR/watcher.log" 2>&1 &
    echo $! > "$PID_DIR/watcher.pid"
    echo "  Watching: vault/_raw-logs/inbox/"
else
    echo "  Inbox watcher already running"
fi

echo ""
echo "✅ Brain is alive!"
echo "   Query:  python -m brain.cli query 'your question'"
echo "   Voice:  python -m brain.cli voice"
echo "   Ingest: python -m brain.cli ingest <file>"
echo "   Stop:   ./scripts/stop.sh"

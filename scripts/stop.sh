#!/usr/bin/env bash
# ── Stop all Digital Brain services ───────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/data"

echo "🧠 Stopping Digital Brain..."

for service in pocketbase watcher; do
    PID_FILE="$PID_DIR/$service.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "  Stopped $service (PID $PID)"
        fi
        rm -f "$PID_FILE"
    fi
done

# Ensure Ollama model is unloaded
if command -v ollama &> /dev/null; then
    curl -sf http://localhost:11434/api/generate \
        -d '{"model":"qwen2.5:3b","keep_alive":0}' > /dev/null 2>&1 || true
    echo "  Ollama model unloaded"
fi

echo "✅ Brain is resting"

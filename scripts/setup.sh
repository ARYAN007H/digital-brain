#!/usr/bin/env bash
# ── Digital Brain Setup Script ────────────────────────────
# One-shot setup: installs deps, pulls model, creates dirs.
# Run from project root: ./scripts/setup.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🧠 Digital Brain — Setup"
echo "========================"

# ── 1. Python virtual environment ─────────────────────────
echo ""
echo "▸ Setting up Python environment..."
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  ✅ Python deps installed"

# ── 2. Ollama + brain model ───────────────────────────────
echo ""
echo "▸ Setting up Ollama..."
if command -v ollama &> /dev/null; then
    echo "  Ollama found"
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

export OLLAMA_KEEP_ALIVE=0
echo "  Pulling qwen2.5:3b (this may take a while)..."
ollama pull qwen2.5:3b
echo "  ✅ Model ready"

# ── 3. PocketBase ─────────────────────────────────────────
echo ""
echo "▸ Setting up PocketBase..."
PB_DIR="$PROJECT_ROOT/bin"
mkdir -p "$PB_DIR"

if [ ! -f "$PB_DIR/pocketbase" ]; then
    echo "  Downloading PocketBase..."
    cd "$PB_DIR"
    wget -q "https://github.com/pocketbase/pocketbase/releases/latest/download/pocketbase_0.25.9_linux_amd64.zip" -O pb.zip
    unzip -qo pb.zip pocketbase
    rm pb.zip
    chmod +x pocketbase
    cd "$PROJECT_ROOT"
    echo "  ✅ PocketBase installed"
else
    echo "  PocketBase already installed"
fi

# ── 4. Piper TTS ─────────────────────────────────────────
echo ""
echo "▸ Setting up Piper TTS..."
if [ ! -f "$PB_DIR/piper" ]; then
    echo "  Downloading Piper..."
    cd "$PB_DIR"
    wget -q "https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz" -O piper.tar.gz
    tar -xzf piper.tar.gz --strip-components=1
    rm piper.tar.gz
    cd "$PROJECT_ROOT"
fi

if [ ! -f "$PB_DIR/en_US-lessac-medium.onnx" ]; then
    echo "  Downloading voice model..."
    cd "$PB_DIR"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    cd "$PROJECT_ROOT"
fi
echo "  ✅ Piper TTS ready"

# ── 5. Environment file ──────────────────────────────────
echo ""
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "▸ Created .env from template. Edit it with your API keys:"
    echo "  $PROJECT_ROOT/.env"
else
    echo "▸ .env already exists"
fi

# ── 6. Directory structure ────────────────────────────────
echo ""
echo "▸ Ensuring directory structure..."
python3 -c "from brain.config import Paths; Paths.ensure_dirs(); print('  ✅ All directories ready')"

# ── Done ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "🧠 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (Groq, Gemini, Supabase)"
echo "  2. Start PocketBase: ./scripts/start.sh"
echo "  3. Open PocketBase admin: http://localhost:8090/_/"
echo "     Create collections: synapse_scores, emotional_tags,"
echo "     pattern_signals, task_events"
echo "  4. Start using: python -m brain.cli query 'hello brain'"
echo "════════════════════════════════════════════"

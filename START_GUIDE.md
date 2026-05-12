# Digital Brain — Start Guide (Manual Setup)

This guide covers **manual configuration** after cloning the repo.

## 1) Install system dependencies (Arch Linux)

```bash
sudo pacman -S --needed python python-pip sqlite git curl unzip wget alsa-utils ffmpeg
```

## 2) Create Python environment and install package

```bash
cd /workspace/digital-brain
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 3) Install Ollama + local model (required for RECALL/CONNECT/DO + fallback)

```bash
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_KEEP_ALIVE=0
ollama pull qwen2.5:3b
```

## 4) Install PocketBase (local synapse/event store)

```bash
mkdir -p bin
cd bin
wget https://github.com/pocketbase/pocketbase/releases/latest/download/pocketbase_linux_amd64.zip
unzip -o pocketbase_linux_amd64.zip
chmod +x pocketbase
cd ..
```

Run once to verify:

```bash
./bin/pocketbase serve
```

Then stop with `Ctrl+C`.

## 5) Install Piper + voice model (optional but needed for TTS)

```bash
cd bin
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
chmod +x piper
cd ..
```

## 6) API keys (manual)

Create `.env` in repo root:

```bash
cat > .env <<'ENV'
# Cloud LLMs (optional but recommended for DECIDE/PREDICT/CREATE)
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key

# Supabase mirror (optional)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_key

# Local services
OLLAMA_HOST=http://localhost:11434
POCKETBASE_URL=http://localhost:8090

# Optional custom paths
# VAULT_ROOT=/absolute/path/to/vault
# DATA_DIR=/absolute/path/to/data
# BIN_DIR=/absolute/path/to/bin
ENV
```

## 7) Start local services

Terminal A:
```bash
ollama serve
```

Terminal B:
```bash
./bin/pocketbase serve
```

Terminal C:
```bash
source .venv/bin/activate
brain watch
```

## 8) First-run checks

```bash
source .venv/bin/activate
brain stats
brain query "RECALL: what do I know about this project?"
```

If you enabled voice:

```bash
brain voice
```

## 9) Common manual fixes

- **No cloud responses for DECIDE/PREDICT/CREATE**: verify `GROQ_API_KEY`/`GEMINI_API_KEY` in `.env`.
- **Voice not audible**: verify `aplay` works and your output device is selected.
- **Local model RAM pressure**: ensure `OLLAMA_KEEP_ALIVE=0` is set and only one Ollama job runs at once.
- **No ingestion from inbox**: confirm files are dropped in `vault/_raw-logs/inbox/` and watcher is running.

## 10) Useful commands

```bash
brain ingest path/to/file.md
brain watch
brain query "DECIDE: should I prioritize X or Y?"
brain surface
brain stats
brain ui
```

## 11) Optional: systemd services

Use provided scripts under `scripts/` to run watcher/nightly jobs as services.

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

(Review scripts before enabling on your machine.)


## 12) Launch the visual web UI

```bash
source .venv/bin/activate
brain ui
```

This opens a local Streamlit app with chat + neuron/synapse views.

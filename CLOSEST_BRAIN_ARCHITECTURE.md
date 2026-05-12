# Closest-Possible Brain-Like Architecture (for i3-1115G4, 8GB RAM)

## Goal
Build the closest practical system to brain-like learning on constrained hardware:
- continuous memory formation
- automatic connection reinforcement
- adaptive retrieval and planning
- realtime visual feedback

## Hardware-aware architecture

```text
Input streams (chat/voice/files)
      ↓
Fast ingestion (local parse + atomic notes)
      ↓
Memory core:
  - Vault markdown neurons (source of truth)
  - Chroma vectors (semantic memory)
  - PocketBase synapses (connection strengths)
  - SQLite history/queue (temporal traces)
      ↓
Plasticity engine (new):
  - reads recent traces
  - reinforces co-activated neuron pairs
  - updates synapse strengths continuously
      ↓
Reasoning router:
  - local qwen for recall/connect/do
  - cloud models for decide/predict/create
      ↓
Realtime UI:
  - 3D graph + click inspect
  - auto refresh + metrics
```

## Why this is closest (practically)
1. **Hebbian-like updates**: co-activity reinforces links.
2. **Multi-store memory**: episodic traces + semantic vectors + symbolic graph.
3. **Online consolidation**: periodic plasticity loop keeps learning alive.
4. **Bounded compute**: no giant always-on model required.

## What was implemented now
- `brain/plasticity.py` with `PlasticityEngine`.
- Reads recent conversation traces from SQLite.
- Detects co-mentioned neuron IDs and reinforces synapses.
- Supports one-shot pass or continuous background loop.

## Run

```bash
# one pass
python -m brain.cli adapt --once

# continuous
python -m brain.cli adapt --interval 300
```

## Next upgrades (recommended)
- spike-timing-like scoring using event timestamps.
- synapse confidence and provenance trails.
- structural growth suggestions (new bridge neurons).
- sleep-phase consolidation job at low CPU windows.


## Implemented improvement (current)
- Plasticity now uses **time-aware weighting** so newer traces reinforce more strongly.
- Added `--dry-run` mode for safe tuning before writing synapses.

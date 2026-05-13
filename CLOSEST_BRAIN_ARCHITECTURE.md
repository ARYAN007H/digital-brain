# Closest-Possible Brain-Like Architecture (for i3-1115G4, 8GB RAM)

## Goal
Build the closest practical system to brain-like learning on constrained hardware:
- continuous memory formation
- automatic connection reinforcement & weakening
- adaptive retrieval with spreading activation
- working memory with topic focus
- autonomous knowledge generation
- sleep-phase consolidation
- emotional modulation
- cross-region integration
- realtime event-driven processing

## Hardware-aware architecture

```text
Input streams (chat/voice/files)
      ↓
Fast ingestion (local parse + atomic notes)
      ↓
      ├──→ Emotional Gate: encoding boost for charged content
      ├──→ Associative Index: tag + FTS5 trigram indexing
      └──→ EventBus: emit neuron.created
      ↓
Memory core:
  - Vault markdown neurons (source of truth)
  - Chroma vectors (semantic memory, all regions)
  - PocketBase synapses (connection strengths)
  - SQLite history/queue (temporal traces)
  - SQLite event_log (neural signals)
  - SQLite working_memory (prefrontal scratchpad)
  - SQLite neuron_access_events (STDP timing)
      ↓
Real-time query pipeline:
  1. Working Memory → primed context (zero-cost)
  2. Associative Recall → tag/fuzzy/temporal hits
  3. Thalamus → cross-region semantic search
  4. Spreading Activation → graph-connected context
  5. Emotional Gate → re-rank by salience
  6. LLM call (local qwen2.5:3b / cloud fallback)
  7. EventBus → neuron.accessed, query.completed
  8. STDP → record access timing for plasticity
  9. Working Memory → update buffer
      ↓
Continuous plasticity:
  - STDP engine: directional reinforcement from access timing
  - Co-mention fallback: bulk data reinforcement
  - Runs every 5 minutes via systemd
      ↓
Nightly brain maintenance (2am):
  Phase 1: Synaptic Decay (LTD + pruning)
  Phase 2: STDP batch pass (24h window)
  Phase 3: Memory Consolidation (replay + bridge + promote)
  Phase 4: Neurogenesis (3 auto-generated insight neurons)
  Phase 5: Associative Index rebuild
  Phase 6: Reports (stats, daily surface, strong synapses, patterns)
  Phase 7: Event log cleanup
      ↓
Realtime UI:
  - 3D graph + click inspect
  - auto refresh + metrics
  - neural subsystem status
```

## Biological mechanisms implemented

| # | Mechanism | Biological Analogue | Module |
|---|-----------|-------------------|--------|
| 1 | **STDP** | Spike-timing dependent plasticity | `brain/stdp.py` |
| 2 | **Synaptic Decay** | Long-term depression + pruning | `brain/decay.py` |
| 3 | **Spreading Activation** | Associative neural propagation | `brain/activation.py` |
| 4 | **Sleep Consolidation** | Hippocampal replay during SWS | `brain/consolidation.py` |
| 5 | **Working Memory** | Prefrontal scratchpad (7±2 items) | `brain/working_memory.py` |
| 6 | **Associative Recall** | Hippocampal pattern completion | `brain/associative.py` |
| 7 | **Neurogenesis** | New neuron generation in hippocampus | `brain/neurogenesis.py` |
| 8 | **Emotional Modulation** | Amygdala gating of encoding/retrieval | `brain/emotional_gate.py` |
| 9 | **Cross-Region Binding** | Thalamocortical integration | `brain/thalamus.py` |
| 10 | **Neural Event Bus** | Action potential signaling backbone | `brain/eventbus.py` |

## CLI commands

```bash
# Core
brain query "what do I know about X?"
brain ingest <file> [--force]
brain watch                    # inbox daemon
brain stats                    # brain health
brain dashboard                # live TUI

# Neural subsystems
brain neural-status            # all subsystem metrics
brain adapt --once             # plasticity pass (STDP + co-mention)
brain decay --preview          # synaptic decay dry-run
brain decay                    # apply decay + prune
brain consolidate              # replay + bridge + promote
brain neurogenesis             # generate insight neurons
brain wm                      # view working memory buffer
brain wm --flush               # clear working memory
brain surface                  # full nightly maintenance
```

## Resource budget (~4.8 GB total)

| Component | RAM |
|-----------|-----|
| OS + iGPU | ~3 GB |
| Ollama qwen2.5:3b (on-demand) | ~2.8 GB |
| ChromaDB | ~150 MB |
| PocketBase | ~50 MB |
| SQLite (brain.db + queue.db) | ~20 MB |
| Event bus + working memory | ~10 MB |
| Watcher daemon | ~30 MB |

"""
Digital Brain CLI — the primary query interface.

Rich terminal UI with panels, tables, progress bars, and color.

Usage:
    brain query "what do I know about X?"
    brain recall "topic"
    brain decide "should I do X or Y?"
    brain ingest <file> [--force]
    brain watch
    brain stats
    brain surface
    brain voice
    brain dashboard
    brain ide [--ingest-existing]
    brain setup-pb
    brain scan-links

Conversation history defaults:
    Enabled, with 30-day retention in data/brain.db.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_console():
    """Get a rich Console, or None if rich not installed."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def cmd_query(args):
    """Route a query through the brain."""
    from brain.router import Router
    from brain.vault import Vault
    from brain.vectors import VectorStore

    console = _get_console()
    router = Router()
    vault = Vault()
    vault.sync_id_counters()

    # Build context from ChromaDB
    context = ""
    try:
        vectors = VectorStore()
        query_text = " ".join(args.query)
        embedding = router.get_embedding(query_text)
        if embedding:
            results = vectors.search(embedding, top_k=3)
            if results:
                context = "\n\n".join(
                    f"[{r['metadata'].get('title', r['id'])}]\n{r['document'][:300]}"
                    for r in results
                )
    except Exception as e:
        logging.warning(f"Context retrieval failed: {e}")

    result = router.route(" ".join(args.query), context=context)

    if console:
        from rich.panel import Panel
        from rich.text import Text
        mode_colors = {
            "RECALL": "cyan", "CONNECT": "blue", "DO": "green",
            "DECIDE": "yellow", "PREDICT": "magenta", "CREATE": "red",
        }
        color = mode_colors.get(result["mode"], "white")
        header = f"[bold {color}]{result['mode']}[/] via [dim]{result['provider']}[/dim]"
        console.print()
        console.print(Panel(
            result["response"],
            title=f"🧠 {header}",
            border_style=color,
            padding=(1, 2),
        ))
        console.print()
    else:
        print(f"\n🧠 [{result['mode']}] via {result['provider']}\n")
        print(result["response"])
        print()


def cmd_ingest(args):
    """Ingest a file into the brain."""
    from brain.ingestion.processor import Processor
    from brain.vault import Vault

    console = _get_console()
    vault = Vault()
    vault.sync_id_counters()
    processor = Processor()

    filepath = Path(args.file).resolve()
    if not filepath.exists():
        msg = f"File not found: {filepath}"
        if console:
            console.print(f"[bold red]✗[/] {msg}")
        else:
            print(msg)
        sys.exit(1)

    force = getattr(args, "force", False)

    if console:
        from rich.progress import Progress, SpinnerColumn, TextColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Ingesting {filepath.name}...", total=None)
            result = processor.process_file(filepath, force=force)
            progress.update(task, completed=True)

        if result.get("status") == "skipped":
            reason = result.get("reason", "unknown")
            console.print(f"[yellow]⏭[/] Skipped: {reason}")
            if reason == "duplicate":
                console.print("[dim]  Use --force to re-ingest[/dim]")
        else:
            console.print(f"[bold green]✅ Ingested:[/] {filepath.name}")
            console.print(f"   Atoms: {result.get('atomic_count', 0)}")
            console.print(f"   Memory: {result.get('memory_id', 'none')}")
            console.print(f"   Tasks: {result.get('task_count', 0)}")
            console.print(f"   Chunks: {result.get('chunks_processed', 0)}")
    else:
        print(f"Ingesting: {filepath.name}")
        result = processor.process_file(filepath, force=force)
        print(f"✅ {result}")


def cmd_watch(args):
    """Start the inbox watcher daemon."""
    from brain.ingestion.watcher import main as watch_main
    watch_main()


def cmd_stats(args):
    """Show brain health stats."""
    from brain.queue import WriteQueue
    from brain.surfacing.nightly import Surfacer
    from brain.synapses import SynapseManager
    from brain.vault import Vault
    from brain.vectors import VectorStore

    console = _get_console()
    vault = Vault()
    vectors = VectorStore()
    synapses = SynapseManager()
    queue = WriteQueue()

    if console:
        from rich.panel import Panel
        from rich.table import Table

        # Neuron table
        counts = vault.count_neurons()
        nt = Table(title="🧠 Neurons", show_header=True, header_style="bold cyan")
        nt.add_column("Region", style="white")
        nt.add_column("Count", justify="right", style="green")
        icons = {"prefrontal": "🎯", "hippocampus": "🧩", "creative": "💡",
                 "predictive": "📊", "amygdala": "❤️", "executive": "✅"}
        for r in ["prefrontal", "hippocampus", "creative", "predictive", "amygdala", "executive"]:
            nt.add_row(f"{icons.get(r, '')} {r}", str(counts.get(r, 0)))
        nt.add_section()
        nt.add_row("[bold]Total[/]", f"[bold yellow]{counts.get('total', 0)}[/]")
        console.print(nt)

        # Vector table
        vc = vectors.count()
        vt = Table(title="🔍 Vectors", show_header=True, header_style="bold magenta")
        vt.add_column("Collection", style="white")
        vt.add_column("Count", justify="right", style="green")
        for r in ["prefrontal", "hippocampus", "creative", "predictive", "executive"]:
            vt.add_row(r, str(vc.get(r, 0)))
        console.print(vt)

        # Queue status
        qc = queue.total_count()
        console.print(Panel(
            f"⏳ Pending: [yellow]{qc['pending']}[/]  "
            f"⚙️ Processing: [cyan]{qc['processing']}[/]  "
            f"✅ Done: [green]{qc['done']}[/]  "
            f"💀 Dead: [red]{qc.get('dead_letter', 0)}[/]",
            title="📡 Sync Queue",
            border_style="blue",
        ))

        # Inbox
        inbox = vault.inbox_count()
        syn = synapses.total_count()
        console.print(f"\n🔗 Synapses: [bold]{syn}[/]  📥 Inbox: [bold]{inbox}[/]\n")
    else:
        surfacer = Surfacer()
        surfacer.generate_brain_stats()
        stats_path = Path("vault/_meta/brain-stats.md")
        if stats_path.exists():
            print(stats_path.read_text())
        else:
            print("No stats yet. Ingest some files first!")


def cmd_surface(args):
    """Run proactive surfacing."""
    from brain.surfacing.nightly import Surfacer
    console = _get_console()
    surfacer = Surfacer()
    surfacer.generate_all()
    if console:
        console.print("[bold green]✅[/] Surfacing complete. Check vault/_meta/")
    else:
        print("✅ Surfacing complete. Check vault/_meta/")


def cmd_voice(args):
    """Start a voice query session."""
    import subprocess
    import tempfile

    from brain.router import Router
    from brain.vault import Vault
    from brain.voice import stt, tts
    from brain.vectors import VectorStore

    vault = Vault()
    vault.sync_id_counters()
    router = Router()

    print("🎤 Voice mode. Press Ctrl+C to exit.")
    print("Recording... (press Enter to stop)")

    try:
        while True:
            # Record audio via arecord
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            input("Press Enter to start recording...")
            print("🔴 Recording... Press Enter to stop.")

            proc = subprocess.Popen(
                ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            input()
            proc.terminate()
            proc.wait()

            # Transcribe
            print("📝 Transcribing...")
            text = stt.transcribe(tmp_path)
            print(f"You said: {text}")

            if not text.strip():
                continue

            # Route query
            context = ""
            try:
                vectors = VectorStore()
                embedding = router.get_embedding(text)
                if embedding:
                    results = vectors.search(embedding, top_k=3)
                    context = "\n\n".join(
                        f"[{r['metadata'].get('title', '')}] {r['document'][:200]}"
                        for r in results
                    )
            except Exception:
                pass

            result = router.route(text, context=context)
            print(f"\n🧠 [{result['mode']}]: {result['response']}\n")

            # Speak response
            tts.speak(result["response"])

            # Log to voice log
            entry = (
                f"**{datetime.now().strftime('%Y-%m-%d %H:%M')}**\n"
                f"Q: {text}\n"
                f"A ({result['provider']}): {result['response'][:500]}"
            )
            vault.write_voice_log(entry)

    except KeyboardInterrupt:
        print("\n👋 Voice mode ended")
        stt.unload()


def cmd_dashboard(args):
    """Launch the live TUI dashboard."""
    from brain.dashboard import run_dashboard
    run_dashboard()


def cmd_ide(args):
    """IDE conversation ingestion."""
    from brain.ingestion.ide_connector import IDEWatcher
    watcher = IDEWatcher()
    if getattr(args, "ingest_existing", False):
        watcher.ingest_existing(since_days=getattr(args, "days", 7))
    else:
        watcher.run_forever()


def cmd_setup_pb(args):
    """Auto-create PocketBase collections."""
    from brain.setup_pocketbase import main as pb_main
    pb_main()


def cmd_scan_links(args):
    """Scan vault for wikilinks and reinforce synapses."""
    from brain.wikilink_scanner import main as scan_main
    scan_main()


def cmd_doctor(args):
    """Run startup/config diagnostics."""
    from brain.security import run_startup_checks

    checks = run_startup_checks()
    has_fail = False
    for c in checks:
        mark = "✅" if c.ok else "❌"
        print(f"{mark} {c.name}: {c.detail}")
        if not c.ok and c.name not in {"groq_key", "gemini_key", "supabase"}:
            has_fail = True

    if has_fail:
        print("\nSome required checks failed. See START_GUIDE.md")



def cmd_ui(args):
    """Launch Streamlit web UI."""
    import subprocess
    import sys
    # Pre-import concurrent.futures.thread to resolve Python 3.14 lazy loading compatibility issues with Streamlit
    cmd = [
        sys.executable, "-c",
        "import concurrent.futures.thread; import sys; from streamlit.web.cli import main; sys.argv=['streamlit', 'run', 'brain/webapp.py']; sys.exit(main())"
    ]
    subprocess.run(cmd, check=False)



def cmd_adapt(args):
    """Run adaptive plasticity updates."""
    from brain.plasticity import PlasticityEngine

    engine = PlasticityEngine()
    if getattr(args, "once", False):
        out = engine.reinforce_from_recent_activity(hours=args.hours, dry_run=args.dry_run)
        print(out)
        return

    engine.run_forever(interval_sec=args.interval)


def cmd_decay(args):
    """Run synaptic decay pass."""
    from brain.decay import DecayEngine

    console = _get_console()
    engine = DecayEngine()

    if getattr(args, "preview", False):
        result = engine.preview_decay()
        label = "Decay preview (dry run)"
    else:
        result = engine.apply_decay()
        label = "Decay applied"

    if console:
        from rich.panel import Panel
        console.print(Panel(
            f"Total synapses: {result['total_synapses']}\n"
            f"Decayed: [yellow]{result['decayed']}[/]\n"
            f"Pruned: [red]{result['pruned']}[/]\n"
            f"Dry run: {result.get('dry_run', False)}",
            title=f"🧹 {label}",
            border_style="yellow",
        ))
    else:
        print(f"{label}: {result}")


def cmd_consolidate(args):
    """Run memory consolidation pass."""
    from brain.consolidation import ConsolidationEngine

    console = _get_console()
    engine = ConsolidationEngine()
    result = engine.run_consolidation()

    if console:
        from rich.panel import Panel
        console.print(Panel(
            f"Replay connections: [green]{result['replay']}[/]\n"
            f"Bridge connections: [cyan]{result['bridges']}[/]\n"
            f"Promotion suggestions: [yellow]{len(result['promotions'])}[/]",
            title="🌙 Consolidation",
            border_style="blue",
        ))
    else:
        print(f"Consolidation: {result}")


def cmd_neurogenesis(args):
    """Run neurogenesis — generate insight neurons."""
    from brain.neurogenesis import NeurogenesisEngine

    console = _get_console()
    engine = NeurogenesisEngine()
    result = engine.run_neurogenesis()

    if console:
        from rich.panel import Panel
        console.print(Panel(
            f"Clusters found: {result['clusters_found']}\n"
            f"Neurons generated: [bold green]{result['neurons_generated']}[/]",
            title="🧬 Neurogenesis",
            border_style="green",
        ))
        for n in result.get("neurons", []):
            console.print(f"  ✨ [[{n['id']}]] {n['title']}")
    else:
        print(f"Neurogenesis: {result}")


def cmd_neural_status(args):
    """Show status of all neural subsystems."""
    console = _get_console()

    sections = []

    # Event bus
    try:
        from brain.eventbus import EventBus
        bus = EventBus.get()
        total = bus.event_count()
        sections.append(("⚡ Event Bus", f"Total events logged: {total}"))
    except Exception as e:
        sections.append(("⚡ Event Bus", f"Error: {e}"))

    # STDP
    try:
        from brain.stdp import STDPEngine
        stdp = STDPEngine()
        count = stdp.access_count()
        sections.append(("🔬 STDP", f"Access events recorded: {count}"))
    except Exception as e:
        sections.append(("🔬 STDP", f"Error: {e}"))

    # Working memory
    try:
        from brain.working_memory import WorkingMemoryBuffer
        wm = WorkingMemoryBuffer()
        size = wm.size()
        focus = wm.get_focus_topic() or "none"
        sections.append(("🧠 Working Memory", f"Buffer size: {size}/7 | Focus: {focus}"))
    except Exception as e:
        sections.append(("🧠 Working Memory", f"Error: {e}"))

    # Decay
    try:
        from brain.decay import DecayEngine
        decay = DecayEngine()
        stats = decay.recent_decay_stats(1)
        if stats:
            last = stats[0]
            sections.append(("🧹 Decay", f"Last run: {last['run_at'][:16]} | Decayed: {last['decayed']} | Pruned: {last['pruned']}"))
        else:
            sections.append(("🧹 Decay", "No decay runs yet"))
    except Exception as e:
        sections.append(("🧹 Decay", f"Error: {e}"))

    # Associative
    try:
        from brain.associative import AssociativeRecallEngine
        assoc = AssociativeRecallEngine()
        idx = assoc.index_stats()
        sections.append(("🔗 Associative Index", f"Tags: {idx['unique_tags']} | Neurons: {idx['tagged_neurons']} | FTS: {idx['fts_indexed']}"))
    except Exception as e:
        sections.append(("🔗 Associative Index", f"Error: {e}"))

    if console:
        from rich.panel import Panel
        from rich.table import Table
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Subsystem", style="white")
        table.add_column("Status", style="green")
        for name, status in sections:
            table.add_row(name, status)
        console.print(Panel(table, title="🧠 Neural Subsystem Status", border_style="magenta"))
    else:
        for name, status in sections:
            print(f"{name}: {status}")


def cmd_wm(args):
    """Working memory operations."""
    from brain.working_memory import WorkingMemoryBuffer

    console = _get_console()
    wm = WorkingMemoryBuffer()

    if getattr(args, "flush", False):
        wm.flush()
        print("Working memory flushed.")
        return

    items = wm.get_active_set()
    if console:
        from rich.table import Table
        table = Table(title="🧠 Working Memory Buffer", show_header=True, header_style="bold cyan")
        table.add_column("Neuron", style="white")
        table.add_column("Region", style="blue")
        table.add_column("Freq", justify="right", style="yellow")
        table.add_column("Relevance", justify="right", style="green")
        for item in items:
            table.add_row(item["neuron_id"], item["region"], str(item["frequency"]), f"{item['relevance']:.3f}")
        console.print(table)
        focus = wm.get_focus_topic()
        if focus:
            console.print(f"\n[dim]Focus topic:[/] {focus}")
    else:
        for item in items:
            print(f"  {item['neuron_id']} ({item['region']}) freq={item['frequency']} rel={item['relevance']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Digital Brain CLI (history defaults: enabled, 30-day retention)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # query
    p_query = sub.add_parser("query", aliases=["q"], help="Query the brain")
    p_query.add_argument("query", nargs="+")
    p_query.set_defaults(func=cmd_query)

    # Shortcut modes
    for mode in ["recall", "connect", "do", "decide", "predict", "create"]:
        p = sub.add_parser(mode, help=f"{mode.upper()} query")
        p.add_argument("query", nargs="+")
        p.set_defaults(func=lambda a, m=mode: cmd_query(
            argparse.Namespace(query=[f"{m.upper()}:"] + a.query, verbose=a.verbose)
        ))

    # ingest
    p_ingest = sub.add_parser("ingest", aliases=["i"], help="Ingest a file")
    p_ingest.add_argument("file")
    p_ingest.add_argument("--force", "-f", action="store_true",
                          help="Force re-ingest even if duplicate")
    p_ingest.set_defaults(func=cmd_ingest)

    # watch
    p_watch = sub.add_parser("watch", aliases=["w"], help="Watch inbox")
    p_watch.set_defaults(func=cmd_watch)

    # stats
    p_stats = sub.add_parser("stats", help="Brain health stats")
    p_stats.set_defaults(func=cmd_stats)

    # surface
    p_surface = sub.add_parser("surface", help="Run nightly brain maintenance")
    p_surface.set_defaults(func=cmd_surface)

    # voice
    p_voice = sub.add_parser("voice", help="Voice query session")
    p_voice.set_defaults(func=cmd_voice)

    # dashboard
    p_dash = sub.add_parser("dashboard", aliases=["dash"],
                            help="Live TUI dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    # web ui
    p_ui = sub.add_parser("ui", help="Launch Streamlit web UI")
    p_ui.set_defaults(func=cmd_ui)

    # ide
    p_ide = sub.add_parser("ide", help="IDE conversation ingestion")
    p_ide.add_argument("--ingest-existing", action="store_true",
                        help="Ingest recent existing conversations")
    p_ide.add_argument("--days", type=int, default=7)
    p_ide.set_defaults(func=cmd_ide)

    # setup-pb
    p_pb = sub.add_parser("setup-pb", help="Auto-create PocketBase collections")
    p_pb.set_defaults(func=cmd_setup_pb)

    # scan-links
    p_links = sub.add_parser("scan-links", help="Scan wikilinks, reinforce synapses")
    p_links.set_defaults(func=cmd_scan_links)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Run startup diagnostics")
    p_doctor.set_defaults(func=cmd_doctor)

    # adapt
    p_adapt = sub.add_parser("adapt", help="Run plasticity adaptation loop")
    p_adapt.add_argument("--once", action="store_true", help="Run one adaptation pass")
    p_adapt.add_argument("--hours", type=int, default=24, help="History window in hours")
    p_adapt.add_argument("--interval", type=int, default=300, help="Loop interval seconds")
    p_adapt.add_argument("--dry-run", action="store_true", help="Compute reinforcement plan without writing")
    p_adapt.set_defaults(func=cmd_adapt)

    # ── New neural subsystem commands ────────────────────

    # decay
    p_decay = sub.add_parser("decay", help="Synaptic decay (LTD + pruning)")
    p_decay.add_argument("--preview", action="store_true", help="Dry-run: show what would decay")
    p_decay.set_defaults(func=cmd_decay)

    # consolidate
    p_consol = sub.add_parser("consolidate", help="Memory consolidation (replay + bridge)")
    p_consol.set_defaults(func=cmd_consolidate)

    # neurogenesis
    p_neuro = sub.add_parser("neurogenesis", aliases=["neuro"], help="Generate insight neurons")
    p_neuro.set_defaults(func=cmd_neurogenesis)

    # neural-status
    p_ns = sub.add_parser("neural-status", aliases=["ns"], help="All neural subsystem metrics")
    p_ns.set_defaults(func=cmd_neural_status)

    # working-memory
    p_wm = sub.add_parser("wm", help="Working memory buffer")
    p_wm.add_argument("--flush", action="store_true", help="Clear working memory")
    p_wm.set_defaults(func=cmd_wm)

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()


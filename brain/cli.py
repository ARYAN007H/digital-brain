"""
Digital Brain CLI — the primary query interface.

Usage:
    brain query "what do I know about X?"
    brain recall "topic"
    brain decide "should I do X or Y?"
    brain ingest <file>
    brain watch
    brain stats
    brain surface
    brain voice
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


def cmd_query(args):
    """Route a query through the brain."""
    from brain.router import Router
    from brain.vault import Vault
    from brain.vectors import VectorStore

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

    print(f"\n🧠 [{result['mode']}] via {result['provider']}\n")
    print(result["response"])
    print()


def cmd_ingest(args):
    """Ingest a file into the brain."""
    from brain.ingestion.processor import Processor
    from brain.vault import Vault

    vault = Vault()
    vault.sync_id_counters()
    processor = Processor()

    filepath = Path(args.file).resolve()
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"Ingesting: {filepath.name}")
    result = processor.process_file(filepath)
    print(f"✅ {result}")


def cmd_watch(args):
    """Start the inbox watcher daemon."""
    from brain.ingestion.watcher import main as watch_main
    watch_main()


def cmd_stats(args):
    """Show brain health stats."""
    from brain.surfacing.nightly import Surfacer
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
    surfacer = Surfacer()
    surfacer.generate_all()
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


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Digital Brain CLI",
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
    p_ingest.set_defaults(func=cmd_ingest)

    # watch
    p_watch = sub.add_parser("watch", aliases=["w"], help="Watch inbox")
    p_watch.set_defaults(func=cmd_watch)

    # stats
    p_stats = sub.add_parser("stats", help="Brain health stats")
    p_stats.set_defaults(func=cmd_stats)

    # surface
    p_surface = sub.add_parser("surface", help="Run proactive surfacing")
    p_surface.set_defaults(func=cmd_surface)

    # voice
    p_voice = sub.add_parser("voice", help="Voice query session")
    p_voice.set_defaults(func=cmd_voice)

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

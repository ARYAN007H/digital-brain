"""
Live TUI dashboard for brain health.

Uses rich.live for real-time terminal display of:
- Neuron counts by region
- Inbox queue status
- Sync status
- Recent activity
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from brain.config import Paths
from brain.queue import WriteQueue
from brain.synapses import SynapseManager
from brain.vault import Vault
from brain.vectors import VectorStore

logger = logging.getLogger(__name__)


def _build_layout():
    """Build the dashboard layout using rich."""
    from rich.layout import Layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="neurons", ratio=2),
        Layout(name="vectors"),
    )
    layout["right"].split_column(
        Layout(name="synapses"),
        Layout(name="queue"),
        Layout(name="inbox"),
    )
    return layout


def _make_header():
    from rich.panel import Panel
    from rich.text import Text
    t = Text("🧠 Digital Brain Dashboard", style="bold white", justify="center")
    return Panel(t, style="bright_blue")


def _make_footer():
    from rich.panel import Panel
    from rich.text import Text
    t = Text(
        f"Last refresh: {datetime.now().strftime('%H:%M:%S')}  |  Press Ctrl+C to exit",
        style="dim", justify="center",
    )
    return Panel(t, style="dim")


def _make_neuron_panel(vault: Vault):
    from rich.panel import Panel
    from rich.table import Table
    counts = vault.count_neurons()

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Region", style="white")
    table.add_column("Count", justify="right", style="green")

    region_icons = {
        "prefrontal": "🎯", "hippocampus": "🧩", "creative": "💡",
        "predictive": "📊", "amygdala": "❤️", "executive": "✅",
    }
    for region in ["prefrontal", "hippocampus", "creative", "predictive", "amygdala", "executive"]:
        icon = region_icons.get(region, "")
        table.add_row(f"{icon} {region}", str(counts.get(region, 0)))

    table.add_section()
    table.add_row("Total", str(counts.get("total", 0)), style="bold yellow")

    return Panel(table, title="[bold]Neurons[/bold]", border_style="cyan")


def _make_vector_panel(vectors: VectorStore):
    from rich.panel import Panel
    from rich.table import Table
    try:
        vcounts = vectors.count()
    except Exception:
        vcounts = {}

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Collection", style="white")
    table.add_column("Vectors", justify="right", style="green")

    total = 0
    for region in ["prefrontal", "hippocampus", "creative", "predictive", "executive"]:
        c = vcounts.get(region, 0)
        total += c
        table.add_row(region, str(c))
    table.add_section()
    table.add_row("Total", str(total), style="bold yellow")

    return Panel(table, title="[bold]Vectors (ChromaDB)[/bold]", border_style="magenta")


def _make_synapse_panel(synapses: SynapseManager):
    from rich.panel import Panel
    from rich.text import Text
    count = synapses.total_count()
    top = synapses.top_synapses(5)

    lines = [f"Total synapses: [bold green]{count}[/bold green]\n"]
    if top:
        lines.append("[bold]Strongest:[/bold]")
        for s in top:
            lines.append(f"  {s.source_id} ↔ {s.target_id} [yellow]({s.strength})[/yellow]")
    else:
        lines.append("[dim]No synapses yet[/dim]")

    from rich.text import Text
    from rich.markup import escape
    content = "\n".join(lines)
    return Panel(content, title="[bold]Synapses[/bold]", border_style="yellow")


def _make_queue_panel(queue: WriteQueue):
    from rich.panel import Panel
    counts = queue.total_count()

    lines = []
    status_styles = {
        "pending": "yellow", "processing": "cyan",
        "done": "green", "failed": "red",
    }
    for status, count in counts.items():
        style = status_styles.get(status, "white")
        icon = {"pending": "⏳", "processing": "⚙️", "done": "✅", "failed": "❌"}.get(status, "•")
        lines.append(f"{icon} {status}: [{style}]{count}[/{style}]")

    return Panel("\n".join(lines), title="[bold]Sync Queue[/bold]", border_style="blue")


def _make_inbox_panel(vault: Vault):
    from rich.panel import Panel
    count = vault.inbox_count()

    if count > 0:
        content = f"[bold yellow]{count}[/bold yellow] files waiting\n\n"
        for f in vault.list_inbox()[:5]:
            content += f"  📄 {f.name}\n"
        if count > 5:
            content += f"  ... and {count - 5} more"
    else:
        content = "[green]✅ Inbox empty[/green]"

    return Panel(content, title="[bold]Inbox[/bold]", border_style="green")


def run_dashboard(refresh_interval: float = 5.0):
    """Run the live dashboard."""
    try:
        from rich.live import Live
    except ImportError:
        print("Dashboard requires 'rich' library: pip install rich")
        return

    vault = Vault()
    vectors = VectorStore()
    synapses_mgr = SynapseManager()
    queue = WriteQueue()

    def make_display():
        layout = _build_layout()
        layout["header"].update(_make_header())
        layout["footer"].update(_make_footer())
        layout["neurons"].update(_make_neuron_panel(vault))
        layout["vectors"].update(_make_vector_panel(vectors))
        layout["synapses"].update(_make_synapse_panel(synapses_mgr))
        layout["queue"].update(_make_queue_panel(queue))
        layout["inbox"].update(_make_inbox_panel(vault))
        return layout

    try:
        with Live(make_display(), refresh_per_second=0.5, screen=True) as live:
            while True:
                time.sleep(refresh_interval)
                live.update(make_display())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_dashboard()

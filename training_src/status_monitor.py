"""Live status monitor for tracking server metrics at runtime."""

import time
import threading
import sys
import os
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn
from rich.box import ROUNDED

# Handle imports from both contexts
try:
    from training_src.tui import get_tui
except ImportError:
    from tui import get_tui


class LiveStatusMonitor:
    """Live status monitor that can run in a separate thread."""

    def __init__(self, refresh_interval: float = 1.0):
        self.refresh_interval = refresh_interval
        self.running = False
        self.console = Console()
        self.tui = get_tui()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the live status monitor."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.console.print(
            "[bold green]✅ Live Status Monitor Started[/bold green]\n"
        )

    def stop(self):
        """Stop the live status monitor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.console.print(
            "\n[bold yellow]⏸️  Live Status Monitor Stopped[/bold yellow]"
        )

    def _run(self):
        """Run the monitor loop."""
        try:
            with Live(self._build_display(), refresh_per_second=1, console=self.console):
                while self.running:
                    time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self.running = False

    def _build_display(self) -> Panel:
        """Build the live display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=2),
        )

        # Header
        title = Text.assemble(
            ("📊 ", "bold cyan"),
            ("SERVER STATUS MONITOR", "bold cyan"),
            (" 📊", "bold cyan"),
        )
        layout["header"].update(Align.center(title))

        # Main content - Status table
        status_table = self.tui.build_status_table()
        layout["main"].update(Panel(status_table, border_style="bold cyan"))

        # Footer with timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        layout["footer"].update(
            Align.center(
                Text(f"Last updated: {now}", style="dim")
            )
        )

        return Panel(layout, expand=True)


def create_status_monitor() -> LiveStatusMonitor:
    """Create a new status monitor."""
    return LiveStatusMonitor()


if __name__ == "__main__":
    monitor = create_status_monitor()

    print("\nStarting live status monitor...")
    print("Press Ctrl+C to stop.\n")

    monitor.start()

    try:
        # Simulate requests
        for i in range(10):
            time.sleep(2)
            monitor.tui.record_request(success=True, tokens=128)
            if i % 3 == 0:
                monitor.tui.record_request(success=False)  # Occasional error

        monitor.stop()
    except KeyboardInterrupt:
        monitor.stop()


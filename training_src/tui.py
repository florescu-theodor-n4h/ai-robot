"""Beautiful TUI for LLM Server status and execution."""
import threading
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.box import ROUNDED

console = Console()


class ServerStatusTUI:
    """Beautiful, colorful TUI for server execution status."""

    def __init__(self):
        self.model_loaded = False
        self.server_running = False
        self.start_time: Optional[datetime] = None
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0

    def show_welcome_banner(self):
        """Show welcome banner with ASCII art."""
        banner = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║       🤖  LLM AGENTIC DEV SERVER  🚀                      ║
  ║                                                           ║
  ║       Powered by TinyLlama + FastAPI + ROCm             ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold cyan")

    def show_startup_progress(self, stage: str, status: str, complete: bool = False):
        """Show startup progress with colors."""
        if complete:
            icon = "✅"
            style = "bold green"
        else:
            icon = "⏳"
            style = "bold yellow"

        msg = f"{icon} {stage:.<40} {status}"
        console.print(msg, style=style)

    def show_error(self, title: str, message: str):
        """Show error message."""
        panel = Panel(
            message,
            title=f"❌ {title}",
            border_style="bold red",
            style="red",
        )
        console.print(panel)

    def show_server_ready(self, config):
        """Show beautiful "server ready" display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=2),
        )

        # Header
        title = Text.assemble(
            ("🚀 ", "bold green"),
            ("SERVER READY FOR TAKEOFF", "bold cyan"),
            (" 🚀", "bold green"),
        )
        layout["header"].update(Align.center(title))

        # Main content
        info_table = Table(show_header=False, box=ROUNDED, padding=(1, 2))

        info_table.add_row(
            Text("🤖 Model", style="bold cyan"),
            Text(config.model.repo, style="bright_yellow"),
        )
        info_table.add_row(
            Text("📡 Server", style="bold cyan"),
            Text(f"{config.server.host}:{config.server.port}", style="bright_green"),
        )
        info_table.add_row(
            Text("🧠 Context Size", style="bold cyan"),
            Text(f"{config.model.ctx_size} tokens", style="bright_magenta"),
        )
        info_table.add_row(
            Text("⚙️  Threads", style="bold cyan"),
            Text(f"{config.model.threads}", style="bright_magenta"),
        )
        info_table.add_row(
            Text("🌡️  Temperature", style="bold cyan"),
            Text(f"{config.server.temperature}", style="bright_magenta"),
        )
        info_table.add_row(
            Text("🔤 Max Tokens", style="bold cyan"),
            Text(f"{config.server.max_tokens}", style="bright_magenta"),
        )

        layout["main"].update(Panel(info_table, border_style="bold green"))

        # Footer with API endpoints
        endpoints = (
            "📍 "
            + " • ".join(
                [
                    f"[bold cyan]/chat[/bold cyan]",
                    f"[bold cyan]/v1/chat/completions[/bold cyan]",
                    f"[bold cyan]/v1/models[/bold cyan]",
                    f"[bold cyan]/health[/bold cyan]",
                ]
            )
        )
        layout["footer"].update(Align.center(endpoints))

        console.print(layout)

    def build_status_table(self) -> Table:
        """Build runtime status table."""
        table = Table(title="📊 Runtime Status", box=ROUNDED, border_style="bold cyan")

        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bright_white")
        table.add_column("Status", justify="center")

        # Uptime
        if self.start_time:
            uptime = datetime.now() - self.start_time
            uptime_str = f"{int(uptime.total_seconds())}s"
            table.add_row(
                "⏱️  Uptime",
                uptime_str,
                "✅" if self.server_running else "❌",
            )

        # Model status
        table.add_row(
            "🤖 Model",
            "Loaded" if self.model_loaded else "Loading",
            "✅" if self.model_loaded else "⏳",
        )

        # Server status
        table.add_row(
            "🔌 Server",
            "Running" if self.server_running else "Starting",
            "✅" if self.server_running else "⏳",
        )

        # Request stats
        table.add_row(
            "📬 Requests",
            str(self.request_count),
            "📈" if self.request_count > 0 else "😴",
        )

        # Error stats
        error_style = "red" if self.error_count > 0 else "green"
        table.add_row(
            "❌ Errors",
            str(self.error_count),
            f"[{error_style}]{'⚠️' if self.error_count > 0 else '✅'}[/{error_style}]",
        )

        # Token stats
        table.add_row(
            "🔤 Tokens Generated",
            f"{self.total_tokens:,}",
            "📊",
        )

        return table

    def display_status_live(self):
        """Display live status updates."""
        with Live(self.build_status_table(), refresh_per_second=1, console=console):
            # This would run in a loop in a real scenario
            pass

    def show_api_examples(self):
        """Show example API calls."""
        examples = """
[bold cyan]📚 QUICK START EXAMPLES[/bold cyan]

[bold yellow]1️⃣  Simple Chat[/bold yellow]
[dim]curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Python?",
    "temperature": 0.6
  }'[/dim]

[bold yellow]2️⃣  OpenAI Compatible[/bold yellow]
[dim]curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.6,
    "max_tokens": 128
  }'[/dim]

[bold yellow]3️⃣  Health Check[/bold yellow]
[dim]curl http://localhost:8888/health[/dim]

[bold yellow]4️⃣  List Models[/bold yellow]
[dim]curl http://localhost:8888/v1/models[/dim]
        """
        panel = Panel(
            examples,
            title="🚀 API Examples",
            border_style="bold green",
            expand=False,
        )
        console.print(panel)

    def show_environment_info(self, config):
        """Show environment configuration."""
        env_table = Table(
            title="⚙️  Environment Configuration",
            box=ROUNDED,
            border_style="bold magenta",
        )

        env_table.add_column("Setting", style="bold magenta")
        env_table.add_column("Value", style="bright_white")

        env_table.add_row("OMP Threads", str(config.env.omp_threads))
        env_table.add_row(
            "Tokenizer Parallel",
            "✅ On" if config.env.tokenizer_parallel else "❌ Off",
        )
        env_table.add_row(
            "ROCm Enabled", "✅ Yes" if config.env.rocm_enabled else "❌ No"
        )
        env_table.add_row(
            "CUDA Enabled", "✅ Yes" if config.env.cuda_enabled else "❌ No"
        )
        env_table.add_row("ROCm GFX Version", config.env.rocm_gfx_version)

        console.print(env_table)

    def record_request(self, success: bool = True, tokens: int = 0):
        """Record a request."""
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.total_tokens += tokens

    def mark_model_loaded(self):
        """Mark model as loaded."""
        self.model_loaded = True

    def mark_server_running(self):
        """Mark server as running."""
        self.server_running = True
        self.start_time = datetime.now()


# Global TUI instance
_tui: Optional[ServerStatusTUI] = None


def get_tui() -> ServerStatusTUI:
    """Get or create TUI instance."""
    global _tui
    if _tui is None:
        _tui = ServerStatusTUI()
    return _tui

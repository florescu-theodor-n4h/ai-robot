"""Tests for TUI module."""

import pytest


class TestServerStatusTUI:
    """Test ServerStatusTUI class."""

    def test_initialization(self, tui):
        """Test TUI initializes with default values."""
        assert tui.model_loaded is False
        assert tui.server_running is False
        assert tui.start_time is None
        assert tui.request_count == 0
        assert tui.error_count == 0
        assert tui.total_tokens == 0

    def test_mark_model_loaded(self, tui):
        """Test marking model as loaded."""
        tui.mark_model_loaded()
        assert tui.model_loaded is True

    def test_mark_server_running(self, tui):
        """Test marking server as running."""
        tui.mark_server_running()
        assert tui.server_running is True
        assert tui.start_time is not None

    def test_record_successful_request(self, tui):
        """Test recording successful request."""
        tui.record_request(success=True, tokens=128)
        assert tui.request_count == 1
        assert tui.error_count == 0
        assert tui.total_tokens == 128

    def test_record_failed_request(self, tui):
        """Test recording failed request."""
        tui.record_request(success=False, tokens=0)
        assert tui.request_count == 1
        assert tui.error_count == 1
        assert tui.total_tokens == 0

    def test_record_multiple_requests(self, tui):
        """Test recording multiple requests."""
        tui.record_request(success=True, tokens=100)
        tui.record_request(success=True, tokens=200)
        tui.record_request(success=False, tokens=0)
        
        assert tui.request_count == 3
        assert tui.error_count == 1
        assert tui.total_tokens == 300

    def test_build_status_table(self, tui):
        """Test building status table."""
        tui.mark_model_loaded()
        tui.mark_server_running()
        tui.record_request(success=True, tokens=128)
        
        table = tui.build_status_table()
        assert table is not None
        # Rich Table objects have rows attribute
        assert hasattr(table, 'rows') or hasattr(table, '_rows')

    def test_show_welcome_banner(self, tui, capsys):
        """Test welcome banner display."""
        tui.show_welcome_banner()
        captured = capsys.readouterr()
        assert "LLM AGENTIC DEV SERVER" in captured.out or "🤖" in captured.out

    def test_show_startup_progress(self, tui, capsys):
        """Test startup progress display."""
        tui.show_startup_progress("Test Stage", "Starting...")
        captured = capsys.readouterr()
        assert "Test Stage" in captured.out

    def test_show_startup_progress_complete(self, tui, capsys):
        """Test startup progress complete display."""
        tui.show_startup_progress("Test Stage", "Done!", complete=True)
        captured = capsys.readouterr()
        assert "Test Stage" in captured.out
        assert "Done!" in captured.out

    def test_show_error(self, tui, capsys):
        """Test error display."""
        tui.show_error("Test Error", "This is a test error message")
        captured = capsys.readouterr()
        assert "Test Error" in captured.out or "error" in captured.out.lower()

    def test_get_tui_singleton(self):
        """Test that get_tui returns singleton instance."""
        from training_src.tui import get_tui
        
        tui1 = get_tui()
        tui2 = get_tui()
        assert tui1 is tui2


class TestTUIIntegration:
    """Integration tests for TUI components."""

    def test_startup_sequence(self, tui, config, capsys):
        """Test complete startup sequence."""
        tui.show_welcome_banner()
        tui.show_startup_progress("Stage 1", "Running...", complete=False)
        tui.show_startup_progress("Stage 1", "Done!", complete=True)
        tui.mark_model_loaded()
        tui.mark_server_running()
        tui.show_server_ready(config)
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_request_tracking_workflow(self, tui):
        """Test complete request tracking workflow."""
        tui.mark_server_running()
        
        # Simulate some requests
        for i in range(5):
            success = i % 2 == 0
            tui.record_request(success=success, tokens=100 + i*10)
        
        assert tui.request_count == 5
        assert tui.error_count == 2
        assert tui.total_tokens == 600  # 100+110+120+130+140

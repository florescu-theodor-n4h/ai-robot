"""LLM Server - Entry point with instrumented logging."""
import signal
import sys
import logging
from typing import Dict, Any

from training_src.logging_setup import setup_logging, get_logger
from training_src.config import setup_environment, get_model_config, get_server_config, SYSTEM_PROMPT
from training_src.model import ModelLoader
from training_src.api import create_app

log = get_logger("main")


def setup_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    log.info("Setting up signal handlers...")

    def shutdown_handler(sig, frame):
        log.warning(f"Received signal {sig}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    log.debug("Signal handlers registered (SIGINT, SIGTERM)")


def build_config() -> Dict[str, Any]:
    """Build complete configuration from all sources."""
    log.info("Building server configuration...")

    setup_environment()
    
    config = {
        "model": get_model_config(),
        "server": get_server_config(),
        "system_prompt": SYSTEM_PROMPT,
    }

    log.info(f"Configuration complete:")
    log.info(f"  Model: {config['model']['repo']}")
    log.info(f"  Server: {config['server']['host']}:{config['server']['port']}")

    return config


def main():
    """Main entry point."""
    # Setup logging first
    setup_logging(level=logging.DEBUG)
    log.info("=" * 60)
    log.info("LLM Server Starting")
    log.info("=" * 60)

    try:
        setup_signal_handlers()
        config = build_config()

        # Load model
        log.info("Initializing model loader...")
        model_cfg = config["model"]
        model_loader = ModelLoader(
            repo_id=model_cfg["repo"],
            filename=model_cfg["file"],
            ctx_size=model_cfg["ctx_size"],
            n_threads=model_cfg["threads"],
            verbose=model_cfg["verbose"],
        )

        log.info("Loading model (this may take a moment)...")
        model_loader.load()

        # Create app
        app = create_app(model_loader, config)

        # Run server
        import uvicorn

        server_cfg = config["server"]
        log.info(f"Starting uvicorn server on {server_cfg['host']}:{server_cfg['port']}")
        log.info("=" * 60)

        uvicorn.run(
            app,
            host=server_cfg["host"],
            port=server_cfg["port"],
            log_level=server_cfg["log_level"],
        )

    except Exception as e:
        log.error(f"Fatal error: {e}")
        log.debug("", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

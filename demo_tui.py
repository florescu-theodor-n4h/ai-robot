"""Demo script showing TUI capabilities without starting the server."""

from training_src.tui import get_tui
from training_src.config import CONFIG


def demo_welcome():
    """Demo the welcome banner."""
    tui = get_tui()
    tui.show_welcome_banner()


def demo_startup():
    """Demo the startup progress indicators."""
    tui = get_tui()

    print("\n" + "=" * 60)
    print("STARTUP PROGRESS SIMULATION")
    print("=" * 60 + "\n")

    tui.show_startup_progress("Environment Setup", "Configuring hardware...")
    tui.show_startup_progress("Environment Setup", "Done!", complete=True)

    tui.show_startup_progress("Model Download", "Fetching TinyLlama...")
    tui.show_startup_progress("Model Download", "Downloaded 4.2GB", complete=True)

    tui.show_startup_progress("Model Initialization", "Loading into memory...")
    tui.show_startup_progress("Model Initialization", "Ready!", complete=True)

    tui.show_startup_progress("API Setup", "Creating FastAPI app...")
    tui.show_startup_progress("API Setup", "Done!", complete=True)

    tui.show_startup_progress("Route Registration", "Registering endpoints...")
    tui.show_startup_progress("Route Registration", "Done!", complete=True)


def demo_ready_screen():
    """Demo the server ready screen."""
    tui = get_tui()
    print("\n" + "=" * 60)
    print("SERVER READY SCREEN")
    print("=" * 60 + "\n")

    tui.mark_model_loaded()
    tui.mark_server_running()
    tui.show_server_ready(CONFIG)


def demo_environment_info():
    """Demo the environment information display."""
    tui = get_tui()
    print("\n" + "=" * 60)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 60 + "\n")

    tui.show_environment_info(CONFIG)


def demo_api_examples():
    """Demo the API examples panel."""
    tui = get_tui()
    print("\n" + "=" * 60)
    print("API QUICK START")
    print("=" * 60 + "\n")

    tui.show_api_examples()


def demo_error():
    """Demo error display."""
    tui = get_tui()
    print("\n" + "=" * 60)
    print("ERROR DISPLAY EXAMPLE")
    print("=" * 60 + "\n")

    tui.show_error(
        "Model Loading Failed",
        "Could not download model from HuggingFace.\n"
        "Please check your internet connection and try again.",
    )


def demo_status_tracking():
    """Demo status tracking."""
    tui = get_tui()
    print("\n" + "=" * 60)
    print("REQUEST TRACKING & STATUS")
    print("=" * 60 + "\n")

    tui.mark_model_loaded()
    tui.mark_server_running()

    # Simulate some requests
    for i in range(5):
        tui.record_request(success=True, tokens=128)

    tui.record_request(success=False)  # One error

    print(tui.build_status_table())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        demo = sys.argv[1]
        if demo == "welcome":
            demo_welcome()
        elif demo == "startup":
            demo_startup()
        elif demo == "ready":
            demo_ready_screen()
        elif demo == "env":
            demo_environment_info()
        elif demo == "api":
            demo_api_examples()
        elif demo == "error":
            demo_error()
        elif demo == "status":
            demo_status_tracking()
        elif demo == "all":
            demo_welcome()
            demo_startup()
            demo_ready_screen()
            demo_environment_info()
            demo_api_examples()
            demo_status_tracking()
        else:
            print(f"Unknown demo: {demo}")
    else:
        print("TUI Demo Script")
        print("===============\n")
        print("Usage: python demo_tui.py [demo_name]\n")
        print("Available demos:")
        print("  welcome  - Show welcome banner")
        print("  startup  - Show startup progress")
        print("  ready    - Show server ready screen")
        print("  env      - Show environment configuration")
        print("  api      - Show API quick start examples")
        print("  error    - Show error display")
        print("  status   - Show request status tracking")
        print("  all      - Run all demos\n")
        print("Examples:")
        print("  python demo_tui.py welcome")
        print("  python demo_tui.py all")

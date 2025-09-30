"""
Data Science Sandbox - Main Application
A gamified data science learning platform

See .github/instructions/.copilot-instructions.md for development guidelines
"""

import argparse

from sandbox.core.game_engine import GameEngine


def main() -> None:
    """Main entry point for the Data Science Sandbox"""
    parser = argparse.ArgumentParser(
        description="Data Science Sandbox - Interactive Learning Platform"
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "cli", "jupyter"],
        default="dashboard",
        help="Choose interface mode",
    )
    parser.add_argument(
        "--level", type=int, choices=range(1, 8), help="Start at specific level (1-7)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset progress and start fresh"
    )

    args = parser.parse_args()

    # Initialize game engine
    game = GameEngine()

    if args.reset:
        game.reset_progress()
        print("✨ Progress reset! Starting fresh...")

    if args.level:
        game.set_current_level(args.level)
        print(f"🎯 Starting at Level {args.level}")

    # Launch interface based on mode
    if args.mode == "dashboard":
        game.launch_dashboard()
    elif args.mode == "cli":
        print("💻 Starting CLI mode...")
        game.start_cli_mode()
    elif args.mode == "jupyter":
        print("📚 Opening Jupyter Lab environment...")
        game.launch_jupyter()


if __name__ == "__main__":
    main()

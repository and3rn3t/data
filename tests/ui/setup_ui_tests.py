#!/usr/bin/env python3
"""
UI Testing Setup Script
Sets up the complete UI testing environment for Data Science Sandbox
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def main() -> int:
    """Main setup function"""
    print("🚀 Setting up UI Testing Framework for Data Science Sandbox")
    print("=" * 60)

    success_count = 0
    total_steps = 0

    # Step 1: Install Python dependencies
    total_steps += 1
    if run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "playwright>=1.40.0",
            "pytest-asyncio>=0.21.0",
            "pytest-html>=4.1.0",
            "requests>=2.31.0",
        ],
        "Installing Python dependencies",
    ):
        success_count += 1

    # Step 2: Install Playwright browsers
    total_steps += 1
    if run_command(
        [sys.executable, "-m", "playwright", "install"],
        "Installing Playwright browsers",
    ):
        success_count += 1

    # Step 3: Create necessary directories
    total_steps += 1
    try:
        ui_test_dir = Path(__file__).parent
        (ui_test_dir / "reports").mkdir(exist_ok=True)
        (ui_test_dir / "screenshots").mkdir(exist_ok=True)
        print("✅ Created test directories successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")

    # Step 4: Validate setup
    total_steps += 1
    if run_command(
        [sys.executable, str(Path(__file__).parent / "test_runner.py"), "--check"],
        "Validating UI testing environment",
    ):
        success_count += 1

    # Step 5: Run setup validation tests
    total_steps += 1
    if run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).parent / "test_setup_validation.py"),
            "-v",
        ],
        "Running setup validation tests",
    ):
        success_count += 1

    print("\n" + "=" * 60)
    print(f"Setup completed: {success_count}/{total_steps} steps successful")

    if success_count == total_steps:
        print("🎉 UI Testing framework is ready!")
        print("\nNext steps:")
        print("1. Run smoke tests: python test_runner.py --smoke")
        print("2. Run full tests: python test_runner.py --full")
        print("3. Generate reports: python test_runner.py --report")
        return 0
    else:
        print("⚠️  Setup completed with some issues")
        print("Check error messages above and retry failed steps")
        return 1


if __name__ == "__main__":
    sys.exit(main())

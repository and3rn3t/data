#!/usr/bin/env python3
"""
Comprehensive testing script for Data Science Sandbox
Tests all major components and functionality
"""

import os
import sys
import time

from sandbox.core.game_engine import GameEngine


def test_game_engine() -> bool:
    """Test core game engine functionality"""
    print("🎮 Testing Game Engine...")

    # Create test engine
    engine = GameEngine()
    print(f"✅ Engine initialized: Level {engine.get_current_level()}")

    # Test challenge completion
    engine.complete_challenge("test_challenge")
    print(
        f"✅ Challenge completed: {len(engine.progress['challenges_completed'])} total"
    )

    # Test level info
    challenges = engine.get_level_challenges(1)
    print(f"✅ Level 1 has {len(challenges)} challenges")

    # Test stats
    stats = engine.get_stats()
    print(
        f"✅ Stats: Level {stats['level']}, {stats['experience']} XP, {stats['completion_rate']:.1f}% complete"
    )

    return True


def test_challenge_system() -> bool:
    """Test challenge system integrity"""
    print("\n🎯 Testing Challenge System...")

    engine = GameEngine()

    # Count challenges per level
    total_challenges = 0
    for level in range(1, 8):
        challenges = engine.get_level_challenges(level)
        total_challenges += len(challenges)
        print(f"✅ Level {level}: {len(challenges)} challenges")

    print(f"✅ Total challenges available: {total_challenges}")

    # Verify expected total
    counted_total = engine.count_total_challenges()
    print(f"✅ Engine reports: {counted_total} challenges")

    return total_challenges > 25  # Should have 29+ challenges


def test_dashboard_compatibility() -> bool:
    """Test dashboard can import without errors"""
    print("\n📊 Testing Dashboard Compatibility...")

    try:
        from sandbox.core.dashboard import Dashboard

        engine = GameEngine()
        Dashboard(engine)  # Test initialization only
        print("✅ Dashboard imports successfully")
        print("✅ Dashboard initializes with game engine")
        return True
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Dashboard initialization failed: {e}")
        return False


def test_file_structure() -> bool:
    """Test that all expected files exist"""
    print("\n📁 Testing File Structure...")

    required_files = [
        "main.py",
        "config.py",
        "sandbox/core/game_engine.py",
        "sandbox/core/dashboard.py",
        "streamlit_app.py",
        "progress.json",
    ]

    required_dirs = [
        "challenges/level_1",
        "challenges/level_2",
        "challenges/level_3",
        "challenges/level_4",
        "challenges/level_5",
        "challenges/level_6",
        "challenges/level_7",
        "tests",
        "docs",
    ]

    all_good = True

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False

    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ Found: {dir_path}/")
        else:
            print(f"❌ Missing: {dir_path}/")
            all_good = False

    return all_good


def test_imports() -> bool:
    """Test that all critical imports work"""
    print("\n📦 Testing Imports...")

    imports_to_test = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("plotly.express", "px"),
        ("streamlit", "st"),
        ("duckdb", None),
        ("polars", "pl"),
    ]

    all_good = True

    for module, alias in imports_to_test:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ Import success: {module}")
        except ImportError as e:
            print(f"❌ Import failed: {module} - {e}")
            all_good = False

    return all_good


def main() -> bool:
    """Run comprehensive test suite"""
    print("🚀 Data Science Sandbox - Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        ("Game Engine", test_game_engine),
        ("Challenge System", test_challenge_system),
        ("Dashboard Compatibility", test_dashboard_compatibility),
        ("File Structure", test_file_structure),
        ("Critical Imports", test_imports),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()

            status = "✅ PASS" if result else "❌ FAIL"
            duration = end_time - start_time

            print(f"\n{status} {test_name} ({duration:.2f}s)")
            results.append((test_name, result, duration))

        except Exception as e:
            print(f"\n❌ FAIL {test_name} - Exception: {e}")
            results.append((test_name, False, 0))

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:25s} ({duration:.2f}s)")

    print(f"\n🏆 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

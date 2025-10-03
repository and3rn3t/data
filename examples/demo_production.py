#!/usr/bin/env python3
"""
Data Science Sandbox - Production Demo Script
Demonstrates the full functionality of the polished system
"""

import time
from typing import Dict, List

from sandbox.core.game_engine import GameEngine


def print_header(title: str) -> None:
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎮 {title}")
    print(f"{'='*60}")


def print_section(title: str) -> None:
    """Print a section header"""
    print(f"\n🔹 {title}")
    print("-" * 40)


def demo_game_engine() -> GameEngine:
    """Demonstrate core game engine functionality"""
    print_header("Data Science Sandbox - Production Demo")

    print("🚀 Initializing enhanced game engine...")
    engine = GameEngine()

    # Show current state
    stats = engine.get_stats()
    print("📊 Current Status:")
    print(f"   Player: {engine.progress['player_name']}")
    print(f"   Level: {stats['level']}/7")
    print(f"   Experience: {stats['experience']} XP")
    print(f"   Completion: {stats['completion_rate']:.1f}%")
    print(f"   Badges: {stats['badges']}")

    return engine


def demo_enhanced_challenges(engine: GameEngine) -> None:
    """Demonstrate enhanced challenge system"""
    print_section("Enhanced Challenge System")

    # Show enhanced challenges for Level 1
    enhanced_challenges = engine.get_enhanced_challenges(1)
    print(f"📚 Level 1 Enhanced Challenges ({len(enhanced_challenges)}):")

    for i, challenge in enumerate(enhanced_challenges[:3], 1):  # Show first 3
        print(f"   {i}. {challenge['title']}")
        print(f"      Difficulty: {challenge.get('difficulty', 'Medium')}")
        print(f"      Time: {challenge.get('estimated_time', '30 minutes')}")
        if challenge.get("description"):
            desc = challenge["description"]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"      Description: {desc}")
        print()


def demo_badge_system(engine: GameEngine) -> None:
    """Demonstrate enhanced badge system"""
    print_section("Enhanced Badge System")

    # Show next available badges
    next_badges = engine.get_next_badges()
    print(f"🏆 Next Available Badges ({len(next_badges)}):")

    for badge in next_badges[:5]:  # Show first 5
        print(f"   🏅 {badge['name']}")
        print(f"      {badge['description']}")
        earned = badge.get("progress", {}).get("earned", False)
        requirements_met = badge.get("progress", {}).get("requirements_met", False)
        status = "✅ Ready to earn!" if requirements_met else "🎯 In progress..."
        print(f"      Status: {status}")
        print()


def demo_challenge_completion(engine: GameEngine) -> None:
    """Demonstrate challenge completion with badge earning"""
    print_section("Challenge Completion with Auto-Badge Earning")

    # Complete a few challenges to trigger badges
    test_challenges = [
        "demo_challenge_1",
        "demo_challenge_2",
        "demo_challenge_3",
        "demo_challenge_4",
        "demo_challenge_5",
    ]

    initial_badges = len(engine.progress["badges_earned"])
    initial_xp = engine.progress["experience_points"]

    print("🎯 Completing demo challenges...")
    for i, challenge in enumerate(test_challenges, 1):
        print(f"\n   Challenge {i}: {challenge}")
        engine.complete_challenge(challenge)
        time.sleep(0.5)  # Brief pause for demo effect

    # Show results
    final_badges = len(engine.progress["badges_earned"])
    final_xp = engine.progress["experience_points"]

    print(f"\n📈 Results:")
    print(f"   XP Gained: {final_xp - initial_xp}")
    print(f"   Badges Earned: {final_badges - initial_badges}")
    print(f"   Total Badges: {final_badges}")

    # Show earned badges
    if final_badges > initial_badges:
        print(f"\n🏆 Newly Earned Badges:")
        recent_badges = engine.progress["badges_earned"][initial_badges:]
        for badge_id in recent_badges:
            if badge_id in engine.badge_manager.badge_triggers:
                badge_info = engine.badge_manager.get_badge_info(badge_id)
                print(f"   🏅 {badge_info.get('name', badge_id)}")


def demo_system_stats(engine: GameEngine) -> None:
    """Show comprehensive system statistics"""
    print_section("System Statistics")

    # Challenge statistics
    total_challenges = engine.count_total_challenges()
    completed_challenges = len(engine.progress["challenges_completed"])

    print(f"📊 Challenge Progress:")
    print(f"   Total Available: {total_challenges}")
    print(f"   Completed: {completed_challenges}")
    print(f"   Completion Rate: {completed_challenges/total_challenges*100:.1f}%")

    # Level progress
    print(f"\n🏆 Level Progress:")
    for level in range(1, 8):
        level_status = engine.progress["level_progress"][str(level)]
        status = (
            "✅"
            if level_status["completed"]
            else "🔓" if level_status["unlocked"] else "🔒"
        )
        print(
            f"   {status} Level {level}: {engine.levels[level]['name'] if hasattr(engine, 'levels') else f'Level {level}'}"
        )

    # Badge progress
    available_badges = len(engine.badge_manager.badge_triggers)
    earned_badges = len(engine.progress["badges_earned"])

    print(f"\n🏅 Badge Progress:")
    print(f"   Available: {available_badges}")
    print(f"   Earned: {earned_badges}")
    print(f"   Completion: {earned_badges/available_badges*100:.1f}%")


def demo_performance_metrics() -> Dict[str, float]:
    """Demonstrate system performance"""
    print_section("Performance Metrics")

    # Test initialization performance
    start_time = time.time()
    engine = GameEngine()
    init_time = time.time() - start_time

    # Test challenge loading performance
    start_time = time.time()
    challenges = engine.get_enhanced_challenges(1)
    load_time = time.time() - start_time

    # Test badge checking performance
    start_time = time.time()
    next_badges = engine.get_next_badges()
    badge_time = time.time() - start_time

    metrics = {
        "initialization": init_time,
        "challenge_loading": load_time,
        "badge_processing": badge_time,
    }

    print("⚡ Performance Results:")
    print(f"   Engine Initialization: {metrics['initialization']:.3f}s")
    print(f"   Challenge Loading: {metrics['challenge_loading']:.3f}s")
    print(f"   Badge Processing: {metrics['badge_processing']:.3f}s")

    return metrics


def main() -> None:
    """Run the complete production demo"""
    try:
        # Core functionality demo
        engine = demo_game_engine()

        # Enhanced features demo
        demo_enhanced_challenges(engine)
        demo_badge_system(engine)
        demo_challenge_completion(engine)
        demo_system_stats(engine)

        # Performance testing
        metrics = demo_performance_metrics()

        # Final summary
        print_header("Demo Complete - Production Ready! 🎉")

        final_stats = engine.get_stats()
        print(f"📊 Final Statistics:")
        print(f"   Level: {final_stats['level']}/7")
        print(f"   Experience: {final_stats['experience']} XP")
        print(f"   Challenges Completed: {final_stats['challenges_completed']}")
        print(f"   Badges Earned: {final_stats['badges']}")
        print(f"   Overall Progress: {final_stats['completion_rate']:.1f}%")

        print(f"\n🚀 System Performance:")
        avg_time = sum(metrics.values()) / len(metrics)
        print(f"   Average Response Time: {avg_time:.3f}s")
        print(
            f"   System Status: {'🟢 Excellent' if avg_time < 0.1 else '🟡 Good' if avg_time < 0.5 else '🔴 Needs Optimization'}"
        )

        print(f"\n✨ Features Demonstrated:")
        print(f"   ✅ Enhanced challenge loading with metadata")
        print(f"   ✅ Automatic badge earning system")
        print(f"   ✅ Real-time progress tracking")
        print(f"   ✅ Performance monitoring")
        print(f"   ✅ Production-ready error handling")

        print(f"\n🎊 The Data Science Sandbox is polished and ready for deployment!")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print(f"   This indicates an issue that needs to be resolved.")
        raise


if __name__ == "__main__":
    main()

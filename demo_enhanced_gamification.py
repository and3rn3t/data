#!/usr/bin/env python3
"""
Enhanced Gamification Demo
Demonstrates the achievement system, auto-validation, and analytics
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sandbox.achievements.challenge_validator import (
    ChallengeHintSystem,
    ChallengeValidator,
)
from sandbox.achievements.enhanced_badge_system import EnhancedBadgeManager
from sandbox.analytics.progress_analytics import (
    LearningAnalytics,
    RecommendationEngine,
    SkillRadarChart,
)
from sandbox.core.enhanced_game_engine import EnhancedGameEngine


def demo_badge_system():
    """Demonstrate the enhanced badge system"""
    print("🏆 ENHANCED BADGE SYSTEM DEMO")
    print("=" * 50)

    badge_manager = EnhancedBadgeManager()

    # Show available badge categories
    print(f"📊 Badge Categories: {list(badge_manager.badge_categories.keys())}")

    # Count total badges
    total_badges = sum(
        len(badges) for badges in badge_manager.badge_categories.values()
    )
    print(f"🎖️ Total Available Badges: {total_badges}")

    # Show some sample badges
    print("\n🎯 Sample Badges by Category:")
    for category, badges in badge_manager.badge_categories.items():
        print(f"\n  {category.upper()} ({len(badges)} badges):")
        for _, badge in list(badges.items())[:3]:  # Show first 3
            print(f"    • {badge['name']}: {badge['description'][:60]}...")

    # Test badge checking for demo user
    user_progress = {
        "challenges_completed": [
            "level_1_challenge_1",
            "level_1_challenge_2",
            "level_1_challenge_3",
            "level_2_challenge_1",
            "level_2_challenge_2",
        ],
        "level": 2,
        "skills": ["python", "pandas"],
        "badges_earned": [],
        "total_xp": 250,
        "completion_scores": [85, 92, 78, 88, 95],
        "skill_scores": {"python": 85, "pandas": 75},
    }

    print("\n🔍 Checking badges for user progress...")
    # Add a completed challenge to demonstrate badge checking
    challenge_data = {
        "id": "level_1_challenge_1",
        "completion_time": 120,
        "score": 85,
        "skills": ["python", "pandas"],
    }
    new_badges = badge_manager.check_and_award_badges(user_progress, challenge_data)

    if new_badges:
        print(f"🎉 New badges earned: {len(new_badges)}")
        for badge_id in new_badges:
            # Find the badge details in all categories
            badge_info = None
            for category_badges in badge_manager.badge_categories.values():
                if badge_id in category_badges:
                    badge_info = category_badges[badge_id]
                    break

            if badge_info:
                print(f"  🏆 {badge_info['name']} (+{badge_info['xp_reward']} XP)")
            else:
                print(f"  🏆 {badge_id}")
    else:
        print("📝 No new badges earned yet - keep learning!")

    return badge_manager


def demo_challenge_validation():
    """Demonstrate the challenge validation system"""
    print("\n🔍 CHALLENGE VALIDATION SYSTEM DEMO")
    print("=" * 50)

    validator = ChallengeValidator()

    # Test different code samples
    test_codes = [
        "print('Hello World!')",  # Simple code
        """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df.mean())
        """,  # Pandas code
        """
import numpy as np
arr = np.array([1, 2, 3])
print(np.mean(arr))
        """,  # NumPy code
        "invalid python syntax!",  # Invalid code
    ]

    print("🔍 Testing Challenge Validation:")
    for i, code in enumerate(test_codes, 1):
        print(f"\n  Test {i}: {code[:30].strip()}...")
        # Create a mock challenge for validation
        mock_challenge = {
            "id": f"test_challenge_{i}",
            "tests": [{"name": "basic_test", "assertion": "True", "points": 10}],
        }
        result = validator.validate_challenge(
            f"test_challenge_{i}", code, mock_challenge
        )
        print(f"    ✅ Success: {result['success']}")
        print(f"    📊 Score: {result.get('score', 0)}/100")
        if result.get("skills_detected"):
            print(f"    🎯 Skills: {result['skills_detected']}")

    # Test hint system
    print("\n💡 Testing Hint System:")
    hint_system = ChallengeHintSystem()
    hint = hint_system.get_hint("level_1_challenge_1", {"attempts": 2}, hint_level=0)
    print(f"   Hint: {hint.get('hint', 'No hint available')}")

    return validator


def demo_analytics_system():
    """Demonstrate the analytics and skill radar system"""
    print("\n📊 ANALYTICS & SKILL RADAR DEMO")
    print("=" * 50)

    analytics = LearningAnalytics()
    skill_radar = SkillRadarChart()
    recommendation_engine = RecommendationEngine()

    # Simulate user activity
    user_id = "demo_user"

    # Generate mock activity data
    print("📈 Generating Learning Analytics...")

    # Simulate skill assessment
    skills = {
        "python": 75,
        "pandas": 65,
        "visualization": 80,
        "machine_learning": 45,
        "statistics": 60,
        "data_cleaning": 70,
    }

    print("🎯 Current Skill Levels:")
    for skill, level in skills.items():
        print(f"  • {skill.replace('_', ' ').title()}: {level}%")

    # Analyze learning patterns
    sample_user_data = {
        "user_id": user_id,
        "challenges_completed": [
            {
                "level": 1,
                "challenge": 1,
                "score": 85,
                "time": 120,
                "topic": "python_basics",
            },
            {
                "level": 1,
                "challenge": 2,
                "score": 92,
                "time": 95,
                "topic": "data_structures",
            },
            {"level": 2, "challenge": 1, "score": 65, "time": 180, "topic": "pandas"},
        ],
        "skills": list(skills.keys()),
        "skill_scores": skills,
        "total_study_time": 300,
        "preferred_difficulty": "medium",
        "learning_history": [],
    }

    analysis = analytics.analyze_learning_patterns(sample_user_data)
    learning_style = analysis.get("learning_style", {})
    print(
        f"\n🧠 Detected Learning Style: {learning_style.get('primary_style', 'Unknown')}"
    )
    print(f"   Confidence: {learning_style.get('confidence', 0):.1f}%")

    # Show improvement areas
    improvement_areas = analysis.get("improvement_areas", [])
    if improvement_areas:
        print("\n🎯 Improvement Areas:")
        for area in improvement_areas[:3]:
            print(
                f"  • {area.get('skill', 'Unknown')}: {area.get('description', 'No description')}"
            )

    # Show strengths
    strengths = analysis.get("strengths", [])
    if strengths:
        print("\n💪 Your Strengths:")
        for strength in strengths[:3]:
            print(
                f"  • {strength.get('skill', 'Unknown')}: {strength.get('description', 'No description')}"
            )

    # Get personalized recommendations
    recommendations = recommendation_engine.get_recommendations(
        sample_user_data, max_recommendations=3
    )
    if recommendations:
        print("\n🚀 Personalized Recommendations:")
        for rec in recommendations:
            print(f"  • {rec.get('title', 'Challenge')}")
            print(f"    Priority: {rec.get('priority', 'Medium')}")
            if rec.get("reason"):
                print(f"    Reason: {rec.get('reason', 'General improvement')}")

    return analytics, skill_radar


def demo_enhanced_game_engine():
    """Demonstrate the integrated enhanced game engine"""
    print("\n🎮 ENHANCED GAME ENGINE INTEGRATION DEMO")
    print("=" * 50)

    enhanced_engine = EnhancedGameEngine()

    # Simulate challenge completion with validation
    print("🚀 Simulating Challenge Completion...")

    challenge_id = "level_1_challenge_1"
    user_code = """
# Simple Python code
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(f"Total: {total}")

# Basic data manipulation
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
print(f"Data: {data}")
"""

    result = enhanced_engine.complete_challenge_with_validation(
        challenge_id, user_code, "demo_user"
    )

    print(f"✅ Challenge Completed: {result['success']}")
    print(f"📊 Validation Score: {result['validation'].get('score', 0)}/100")
    print(f"🏆 Badges Earned: {len(result.get('badges_earned', []))}")
    print(f"💰 Total XP Gained: {result.get('total_xp', 0)}")

    if result.get("badges_earned"):
        for badge in result["badges_earned"]:
            print(f"  🎖️ {badge['name']}: +{badge['xp_reward']} XP")

    return enhanced_engine


def main():
    """Run the complete enhanced gamification demo"""
    print("🎮 DATA SCIENCE SANDBOX - ENHANCED GAMIFICATION DEMO")
    print("=" * 60)
    print("Demonstrating achievement system, auto-validation, and analytics")
    print("=" * 60)

    try:
        # Demo each component
        demo_badge_system()
        demo_challenge_validation()
        demo_analytics_system()
        demo_enhanced_game_engine()

        print("\n🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("✅ All enhanced gamification systems are working correctly!")
        print("🚀 Ready to launch enhanced learning experience!")

        return True

    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

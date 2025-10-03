import sys

sys.path.append(".")
from sandbox.core.game_engine import GameEngine

engine = GameEngine()

print("Level 6: Data Science Master - Challenge Status")
print("=" * 50)

challenges = engine.get_level_challenges(6)
completed_ids = [
    c for c in engine.progress["challenges_completed"] if c.startswith("level_6")
]

print("Current Level 6 Challenges:")
for i, challenge in enumerate(challenges, 1):
    is_completed = engine._is_challenge_completed(6, challenge)
    status = "COMPLETED" if is_completed else "MISSING"
    print(f"{i}. {challenge} - {status}")

print("\nCompleted Level 6 IDs in progress:")
for challenge_id in completed_ids:
    print(f"  {challenge_id}")

stats = engine.get_level_completion_stats(6)
print("\nLevel 6 Stats:")
print(
    f'Completed: {stats["completed_challenges"]}/{stats["total_challenges"]} ({stats["completion_rate"]:.1f}%)'
)
needed = 6 - stats["completed_challenges"]  # Need 6 out of 7 for 85%+
print(f"Need {needed} more challenges for 80%+ completion")

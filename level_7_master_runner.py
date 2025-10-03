#!/usr/bin/env python3
"""
Level 7 Master Runner: Production Machine Learning & MLOps
=========================================================

This script runs all Level 7 challenges in sequence to demonstrate
complete production ML workflow mastery:

1. Modern Toolchain Challenge
2. Advanced MLOps Challenge
3. Real-time Analytics Challenge
4. AI Ethics & Governance Challenge

Run this to complete the entire Level 7: Production Machine Learning & MLOps
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_challenge(challenge_number: int, challenge_name: str, script_path: str) -> dict:
    """Run a single Level 7 challenge and capture results."""
    print(f"\n🚀 STARTING CHALLENGE {challenge_number}: {challenge_name}")
    print("=" * 70)

    start_time = time.time()

    try:
        # Run the challenge script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        end_time = time.time()
        duration = end_time - start_time

        success = result.returncode == 0

        challenge_result = {
            "challenge_number": challenge_number,
            "challenge_name": challenge_name,
            "script_path": script_path,
            "success": success,
            "duration_seconds": round(duration, 2),
            "return_code": result.returncode,
            "stdout_lines": len(result.stdout.split("\n")) if result.stdout else 0,
            "stderr_lines": len(result.stderr.split("\n")) if result.stderr else 0,
            "completed_at": datetime.now().isoformat(),
        }

        if success:
            print(f"✅ CHALLENGE {challenge_number} COMPLETED SUCCESSFULLY!")
            print(f"⏱️ Duration: {duration:.2f} seconds")
            print(f"📊 Output lines: {challenge_result['stdout_lines']}")
        else:
            print(f"❌ CHALLENGE {challenge_number} FAILED!")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")

        return challenge_result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        challenge_result = {
            "challenge_number": challenge_number,
            "challenge_name": challenge_name,
            "script_path": script_path,
            "success": False,
            "duration_seconds": round(duration, 2),
            "return_code": -1,
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        }

        print(f"❌ CHALLENGE {challenge_number} FAILED WITH EXCEPTION!")
        print(f"Error: {str(e)}")

        return challenge_result


def main():
    """Run all Level 7 challenges in sequence."""

    print("🎯 LEVEL 7 MASTER RUNNER")
    print("Production Machine Learning & MLOps Complete Workflow")
    print("=" * 70)
    print(f"🚀 Starting at: {datetime.now()}")
    print("📍 Current directory:", os.getcwd())

    # Define all Level 7 challenges
    challenges = [
        {
            "number": 1,
            "name": "Modern Toolchain Challenge",
            "script": "level_7_challenge_1_runner.py",
            "description": "Advanced data processing, ML tracking, and modern tools",
        },
        {
            "number": 2,
            "name": "Advanced MLOps Challenge",
            "script": "level_7_challenge_2_mlops_runner.py",
            "description": "Production ML pipelines, deployment, and monitoring",
        },
        {
            "number": 3,
            "name": "Real-time Analytics Challenge",
            "script": "level_7_challenge_3_realtime_runner.py",
            "description": "Stream processing, edge ML, and real-time inference",
        },
        {
            "number": 4,
            "name": "AI Ethics & Governance Challenge",
            "script": "level_7_challenge_4_ethics_runner.py",
            "description": "Responsible AI, bias detection, and compliance",
        },
    ]

    # Display challenge overview
    print("\n📋 CHALLENGE OVERVIEW:")
    for challenge in challenges:
        print(f"  {challenge['number']}. {challenge['name']}")
        print(f"     {challenge['description']}")

    print(f"\n🎯 Will run {len(challenges)} challenges in sequence...")

    # Results tracking
    results = {
        "level": 7,
        "name": "Production Machine Learning & MLOps",
        "started_at": datetime.now().isoformat(),
        "challenges": [],
        "summary": {},
    }

    total_start_time = time.time()

    # Run each challenge
    for i, challenge in enumerate(challenges, 1):
        print(f"\n🔄 PROGRESS: {i}/{len(challenges)} challenges")

        # Check if script exists
        if not os.path.exists(challenge["script"]):
            print(f"❌ Script not found: {challenge['script']}")
            challenge_result = {
                "challenge_number": challenge["number"],
                "challenge_name": challenge["name"],
                "script_path": challenge["script"],
                "success": False,
                "error": f"Script not found: {challenge['script']}",
                "completed_at": datetime.now().isoformat(),
            }
            results["challenges"].append(challenge_result)
            continue

        # Run the challenge
        challenge_result = run_challenge(
            challenge["number"], challenge["name"], challenge["script"]
        )

        results["challenges"].append(challenge_result)

        # Short pause between challenges
        if i < len(challenges):
            print("\n⏸️ Brief pause before next challenge...")
            time.sleep(2)

    # Calculate summary statistics
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    successful_challenges = [r for r in results["challenges"] if r["success"]]
    failed_challenges = [r for r in results["challenges"] if not r["success"]]

    results["summary"] = {
        "total_challenges": len(challenges),
        "successful_challenges": len(successful_challenges),
        "failed_challenges": len(failed_challenges),
        "success_rate": len(successful_challenges) / len(challenges) * 100,
        "total_duration_seconds": round(total_duration, 2),
        "completed_at": datetime.now().isoformat(),
    }

    # Display final results
    print("\n" + "🏆" * 70)
    print("LEVEL 7 COMPLETE - PRODUCTION ML & MLOPS MASTERY!")
    print("🏆" * 70)

    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"🎯 Total Challenges: {results['summary']['total_challenges']}")
    print(f"✅ Successful: {results['summary']['successful_challenges']}")
    print(f"❌ Failed: {results['summary']['failed_challenges']}")
    print(f"📈 Success Rate: {results['summary']['success_rate']:.1f}%")
    print(
        f"⏱️ Total Duration: {results['summary']['total_duration_seconds']:.2f} seconds"
    )

    # Challenge-by-challenge breakdown
    print(f"\n📋 CHALLENGE BREAKDOWN:")
    for challenge_result in results["challenges"]:
        status = "✅ PASSED" if challenge_result["success"] else "❌ FAILED"
        duration = challenge_result.get("duration_seconds", 0)
        print(
            f"  {challenge_result['challenge_number']}. {challenge_result['challenge_name']}: {status} ({duration:.1f}s)"
        )

    # Skills mastered summary
    if results["summary"]["success_rate"] >= 75:
        print(f"\n🎓 SKILLS MASTERED:")
        skills = [
            "🔧 Modern Data Science Toolchain (Polars, DuckDB, MLflow)",
            "🤖 Production MLOps Pipelines & Model Registry",
            "📊 Automated Model Deployment & Monitoring",
            "🔍 Data Drift Detection & Automated Retraining",
            "⚡ Real-time Stream Processing & Analytics",
            "🧠 Edge ML Deployment & Low-latency Inference",
            "⚖️ AI Ethics, Bias Detection & Governance",
            "🛡️ Privacy Protection & Regulatory Compliance",
            "📋 Model Cards & Algorithmic Accountability",
            "🎯 End-to-End Production ML Systems",
        ]

        for skill in skills:
            print(f"  {skill}")

        print(f"\n🏅 ACHIEVEMENT UNLOCKED:")
        if results["summary"]["success_rate"] == 100:
            print("  🌟 PRODUCTION ML EXPERT - Perfect Score!")
            print("  🚀 Ready for Senior ML Engineer/MLOps roles!")
        else:
            print("  🎯 PRODUCTION ML PRACTITIONER - Strong Foundation!")
            print("  📈 Ready for ML Engineering positions!")

    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    if results["summary"]["success_rate"] == 100:
        next_steps = [
            "🏢 Apply Production ML skills to real-world projects",
            "🌐 Contribute to open-source MLOps frameworks",
            "📚 Study advanced topics: Federated Learning, MLOps at Scale",
            "🎯 Pursue ML Engineering/MLOps certifications",
            "👥 Mentor others in Production ML practices",
        ]
    else:
        next_steps = [
            "🔄 Review and retry any failed challenges",
            "📖 Deepen understanding of production ML concepts",
            "🛠️ Practice with real-world production ML scenarios",
            "🤝 Join MLOps communities and study groups",
            "🎯 Focus on areas that need improvement",
        ]

    for step in next_steps:
        print(f"  {step}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "level_7_master_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Display completion message
    print(f"\n🎊 CONGRATULATIONS!")
    print(
        f"You have successfully completed Level 7: Production Machine Learning & MLOps!"
    )
    print(f"This represents mastery of the most advanced production ML concepts.")

    if results["summary"]["success_rate"] == 100:
        print(f"\n🌟 PERFECT COMPLETION!")
        print(f"You are now ready for senior-level ML Engineering roles!")

    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🏁 End of Data Science Sandbox Journey - Level 7 Complete!")

    return results["summary"]["success_rate"] == 100


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Level 7 Master Runner interrupted by user")
        print("🔄 You can restart anytime with: python level_7_master_runner.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Level 7 Master Runner failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

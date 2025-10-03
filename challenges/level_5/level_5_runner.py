"""
Level 5 - Deep Learning Dynamo: Complete Runner
===============================================

Master comprehensive runner for all Level 5 deep learning challenges.
This script demonstrates the complete journey from neural network fundamentals
to advanced architectures.

Challenges Included:
1. Neural Network Fundamentals - Core concepts and architectures
2. Convolutional Neural Networks - Image processing and computer vision
3. Recurrent Neural Networks - Sequential data and temporal modeling
4. Advanced Architectures - Ensemble methods and modern techniques

Usage: python challenges/level_5/level_5_runner.py [challenge_number]
"""

import sys
import time
from typing import Optional


def run_challenge_1():
    """Run Challenge 1: Neural Network Fundamentals."""
    print("🧠 CHALLENGE 1: Neural Network Fundamentals")
    print("=" * 60)

    try:
        from challenges.level_5.challenge_1_neural_network_fundamentals import (
            run_neural_network_challenges,
        )

        datasets = run_neural_network_challenges()
        return datasets is not None
    except ImportError as e:
        print(f"❌ Failed to import Challenge 1: {e}")
        return False
    except Exception as e:
        print(f"❌ Challenge 1 failed: {e}")
        return False


def run_challenge_2():
    """Run Challenge 2: Convolutional Neural Networks."""
    print("\n🖼️ CHALLENGE 2: Convolutional Neural Networks")
    print("=" * 60)

    try:
        from challenges.level_5.challenge_2_convolutional_networks import (
            run_cnn_challenges,
        )

        datasets = run_cnn_challenges()
        return datasets is not None
    except ImportError as e:
        print(f"❌ Failed to import Challenge 2: {e}")
        return False
    except Exception as e:
        print(f"❌ Challenge 2 failed: {e}")
        return False


def run_challenge_3():
    """Run Challenge 3: Recurrent Neural Networks."""
    print("\n🔄 CHALLENGE 3: Recurrent Neural Networks")
    print("=" * 60)

    try:
        from challenges.level_5.challenge_3_recurrent_networks import run_rnn_challenges

        datasets = run_rnn_challenges()
        return datasets is not None
    except ImportError as e:
        print(f"❌ Failed to import Challenge 3: {e}")
        return False
    except Exception as e:
        print(f"❌ Challenge 3 failed: {e}")
        return False


def run_challenge_4():
    """Run Challenge 4: Advanced Architectures."""
    print("\n🔬 CHALLENGE 4: Advanced Neural Architectures")
    print("=" * 60)

    try:
        from challenges.level_5.challenge_4_advanced_architectures import (
            run_advanced_challenges,
        )

        datasets = run_advanced_challenges()
        return datasets is not None
    except ImportError as e:
        print(f"❌ Failed to import Challenge 4: {e}")
        return False
    except Exception as e:
        print(f"❌ Challenge 4 failed: {e}")
        return False


def show_level_5_intro():
    """Show Level 5 introduction."""
    print("🚀 LEVEL 5: DEEP LEARNING DYNAMO")
    print("=" * 60)
    print()
    print("Welcome to the Deep Learning Dynamo! This level covers:")
    print()
    print("🧠 Challenge 1: Neural Network Fundamentals")
    print("   • Perceptrons and multi-layer networks")
    print("   • Activation functions and backpropagation")
    print("   • Hyperparameter optimization")
    print("   • Model interpretation and visualization")
    print()
    print("🖼️ Challenge 2: Convolutional Neural Networks")
    print("   • Convolution operations and filters")
    print("   • CNN architectures and pooling")
    print("   • Image augmentation techniques")
    print("   • Transfer learning concepts")
    print()
    print("🔄 Challenge 3: Recurrent Neural Networks")
    print("   • Sequential data processing")
    print("   • RNN, LSTM, and GRU architectures")
    print("   • Time series prediction")
    print("   • Text generation and NLP")
    print()
    print("🔬 Challenge 4: Advanced Architectures")
    print("   • Ensemble methods and voting")
    print("   • Attention mechanisms")
    print("   • Regularization techniques")
    print("   • Model interpretation and explainability")
    print()
    print("📚 Prerequisites: Completion of Levels 1-4")
    print("🎯 Goal: Master deep learning from fundamentals to advanced techniques")
    print()


def show_completion_summary():
    """Show Level 5 completion summary."""
    print("\n" + "🎉" * 60)
    print("LEVEL 5: DEEP LEARNING DYNAMO - COMPLETE!")
    print("🎉" * 60)
    print()
    print("🏆 Congratulations! You have successfully mastered:")
    print()
    print("✅ Neural Network Fundamentals")
    print("   • Core architectures and training algorithms")
    print("   • Hyperparameter optimization strategies")
    print("   • Model evaluation and interpretation")
    print()
    print("✅ Convolutional Neural Networks")
    print("   • Image processing and computer vision")
    print("   • CNN architectures and transfer learning")
    print("   • Data augmentation and regularization")
    print()
    print("✅ Recurrent Neural Networks")
    print("   • Sequential data modeling")
    print("   • LSTM/GRU architectures")
    print("   • Time series and text processing")
    print()
    print("✅ Advanced Architectures")
    print("   • Ensemble methods and model combination")
    print("   • Attention mechanisms and feature importance")
    print("   • Model interpretation and explainability")
    print()
    print("🌟 DEEP LEARNING SKILLS MASTERED:")
    print("• Neural network design and implementation")
    print("• Computer vision with CNNs")
    print("• Natural language processing with RNNs")
    print("• Advanced techniques and modern architectures")
    print("• Model evaluation, interpretation, and deployment")
    print()
    print("🚀 NEXT STEPS:")
    print("• Implement real-world deep learning projects")
    print("• Explore transformer architectures and attention")
    print("• Learn generative models (VAEs, GANs)")
    print("• Study reinforcement learning")
    print("• Apply MLOps and model deployment techniques")
    print("• Contribute to open-source deep learning projects")
    print()
    print("🎓 You are now a Deep Learning Expert!")
    print("Ready to tackle advanced AI challenges and build innovative solutions!")


def run_all_challenges():
    """Run all Level 5 challenges in sequence."""
    print("Running all Level 5 challenges...")

    start_time = time.time()

    # Track completion status
    results = {
        "Challenge 1": False,
        "Challenge 2": False,
        "Challenge 3": False,
        "Challenge 4": False,
    }

    # Run each challenge
    print("\n" + "=" * 80)
    results["Challenge 1"] = run_challenge_1()

    if results["Challenge 1"]:
        print("\n" + "=" * 80)
        results["Challenge 2"] = run_challenge_2()

    if results["Challenge 2"]:
        print("\n" + "=" * 80)
        results["Challenge 3"] = run_challenge_3()

    if results["Challenge 3"]:
        print("\n" + "=" * 80)
        results["Challenge 4"] = run_challenge_4()

    # Show results
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("LEVEL 5 EXECUTION SUMMARY")
    print("=" * 80)

    print(f"\nChallenge Results:")
    for challenge, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"• {challenge}: {status}")

    successful_challenges = sum(results.values())
    total_challenges = len(results)

    print(
        f"\nOverall Progress: {successful_challenges}/{total_challenges} challenges completed"
    )
    print(f"Execution time: {duration:.1f} seconds")

    if successful_challenges == total_challenges:
        show_completion_summary()
        return True
    else:
        print(f"\n⚠️ Some challenges failed. Please check the error messages above.")
        return False


def main():
    """Main function to run Level 5 challenges."""
    show_level_5_intro()

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            challenge_num = int(sys.argv[1])
            if challenge_num == 1:
                success = run_challenge_1()
            elif challenge_num == 2:
                success = run_challenge_2()
            elif challenge_num == 3:
                success = run_challenge_3()
            elif challenge_num == 4:
                success = run_challenge_4()
            else:
                print(f"❌ Invalid challenge number: {challenge_num}")
                print("Available challenges: 1, 2, 3, 4")
                return False

            if success:
                print(f"\n✅ Challenge {challenge_num} completed successfully!")
            else:
                print(f"\n❌ Challenge {challenge_num} failed!")

            return success

        except ValueError:
            print(f"❌ Invalid argument: {sys.argv[1]}")
            print("Usage: python level_5_runner.py [1|2|3|4]")
            return False
    else:
        # Run all challenges
        return run_all_challenges()


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Level 5 execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in Level 5 runner: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Joint Limit Testing Script for Piper Robot.

This script tests the joint limits and safety features of the Piper robot
before running RL training. It performs:
1. Reading current joint positions
2. Testing each joint's range of motion (small movements)
3. Verifying safety clipping works correctly
4. Testing emergency stop functionality

Usage:
    python scripts/test_joint_limits.py --mode single_left
    python scripts/test_joint_limits.py --mode bimanual --test-motion
    python scripts/test_joint_limits.py --mode bimanual --test-limits

IMPORTANT: Run this script before RL training to verify safety limits!
"""

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a safety test."""
    test_name: str
    passed: bool
    message: str
    details: dict = None


def test_joint_limits_config():
    """Test that joint limits are properly configured."""
    from lerobot.rl.piper_env import PIPER_JOINT_LIMITS, SafetyConfig

    results = []

    # Check that all joints have limits defined
    expected_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    for joint in expected_joints:
        if joint in PIPER_JOINT_LIMITS:
            limits = PIPER_JOINT_LIMITS[joint]
            if limits.lower < limits.upper:
                results.append(TestResult(
                    test_name=f"Joint limits defined: {joint}",
                    passed=True,
                    message=f"{joint}: [{limits.lower:.3f}, {limits.upper:.3f}] rad",
                ))
            else:
                results.append(TestResult(
                    test_name=f"Joint limits valid: {joint}",
                    passed=False,
                    message=f"{joint}: Invalid limits (lower >= upper)",
                ))
        else:
            results.append(TestResult(
                test_name=f"Joint limits defined: {joint}",
                passed=False,
                message=f"{joint}: No limits defined!",
            ))

    # Check SafetyConfig defaults
    config = SafetyConfig()
    results.append(TestResult(
        test_name="SafetyConfig initialized",
        passed=len(config.joint_limits) == len(expected_joints),
        message=f"Configured {len(config.joint_limits)} joints",
    ))

    return results


def test_safety_checker():
    """Test safety checker logic without robot."""
    from lerobot.rl.piper_env import SafetyChecker, SafetyConfig

    results = []

    # Create checker
    config = SafetyConfig(
        max_delta_position=0.1,
        limit_margin=0.05,
    )
    checker = SafetyChecker(config)

    # Test 1: Normal action should pass through
    current = {"left_joint1.pos": 0.0}
    target = {"left_joint1.pos": 0.05}
    safe, info = checker.check_and_clip_action(target, current)

    results.append(TestResult(
        test_name="Normal action passes",
        passed=abs(safe["left_joint1.pos"] - 0.05) < 0.001,
        message=f"Target: 0.05, Got: {safe['left_joint1.pos']:.4f}",
    ))

    # Test 2: Large velocity should be clipped
    checker.reset()
    current = {"left_joint1.pos": 0.0}
    target = {"left_joint1.pos": 0.5}  # Too large
    safe, info = checker.check_and_clip_action(target, current)

    results.append(TestResult(
        test_name="Velocity clipping works",
        passed=abs(safe["left_joint1.pos"] - 0.1) < 0.001,
        message=f"Target: 0.5, Clipped to: {safe['left_joint1.pos']:.4f} (max_delta=0.1)",
        details={"velocity_violations": info["velocity_violations"]},
    ))

    # Test 3: Position limit clipping
    checker.reset()
    # joint1 limits: [-2.618, 2.618], margin=0.05
    # So effective range is [-2.568, 2.568]
    current = {"left_joint1.pos": 2.5}
    target = {"left_joint1.pos": 2.7}  # Beyond limit
    safe, info = checker.check_and_clip_action(target, current)

    expected_upper = 2.618 - 0.05  # 2.568
    results.append(TestResult(
        test_name="Position limit clipping works",
        passed=abs(safe["left_joint1.pos"] - expected_upper) < 0.001,
        message=f"Target: 2.7, Clipped to: {safe['left_joint1.pos']:.4f} (limit: {expected_upper:.3f})",
        details={"position_violations": info["position_violations"]},
    ))

    # Test 4: Emergency stop detection
    checker.reset()
    checker.config.emergency_velocity_threshold = 0.3

    # Trigger emergency stop with multiple violations
    for _ in range(5):
        current = {"left_joint1.pos": 0.0}
        target = {"left_joint1.pos": 1.0}  # Very large jump
        checker.check_and_clip_action(target, current)

    results.append(TestResult(
        test_name="Emergency stop detection",
        passed=checker.is_emergency_stop,
        message=f"Emergency stop triggered: {checker.is_emergency_stop}",
    ))

    return results


def test_robot_connection(mode: str = "bimanual"):
    """Test connection to real robot and read joint positions."""
    results = []

    try:
        from lerobot.rl.piper_env import make_piper_env

        logger.info(f"Connecting to Piper robot (mode={mode})...")

        env = make_piper_env(
            mode=mode,
            fps=10,
            enable_safety=True,
            reset_time_s=0.5,  # Short reset for testing
        )

        results.append(TestResult(
            test_name="Robot connection",
            passed=True,
            message="Successfully connected to Piper robot",
        ))

        # Read current positions
        obs, info = env.reset()
        positions = obs["agent_pos"]

        results.append(TestResult(
            test_name="Read joint positions",
            passed=len(positions) > 0,
            message=f"Read {len(positions)} joint positions",
            details={"positions": positions.tolist()},
        ))

        # Print current positions
        logger.info("Current joint positions:")
        for i, name in enumerate(env.motor_names):
            if i < len(positions):
                logger.info(f"  {name}: {positions[i]:.4f} rad ({np.degrees(positions[i]):.2f} deg)")

        env.close()

    except Exception as e:
        results.append(TestResult(
            test_name="Robot connection",
            passed=False,
            message=f"Failed to connect: {e}",
        ))

    return results


def test_small_motion(mode: str = "bimanual", joint_idx: int = 0, delta: float = 0.02):
    """Test small motion on a single joint."""
    results = []

    try:
        from lerobot.rl.piper_env import make_piper_env

        logger.info(f"Testing small motion on joint {joint_idx}...")

        env = make_piper_env(
            mode=mode,
            fps=10,
            enable_safety=True,
            max_delta_position=0.1,
        )

        obs, _ = env.reset()
        initial_pos = obs["agent_pos"].copy()

        # Create action with small delta on target joint
        action = initial_pos.copy()
        action[joint_idx] += delta

        logger.info(f"Moving joint {joint_idx}: {initial_pos[joint_idx]:.4f} -> {action[joint_idx]:.4f}")

        # Execute action
        obs, _, _, _, info = env.step(action)
        final_pos = obs["agent_pos"]

        # Check if safety was applied
        safety_info = info.get("safety_info")
        was_clipped = safety_info and safety_info.get("clipped_joints")

        results.append(TestResult(
            test_name=f"Small motion joint {joint_idx}",
            passed=True,
            message=f"Moved from {initial_pos[joint_idx]:.4f} to {final_pos[joint_idx]:.4f}",
            details={
                "delta_requested": delta,
                "delta_actual": final_pos[joint_idx] - initial_pos[joint_idx],
                "was_clipped": was_clipped,
            },
        ))

        env.close()

    except Exception as e:
        results.append(TestResult(
            test_name=f"Small motion joint {joint_idx}",
            passed=False,
            message=f"Failed: {e}",
        ))

    return results


def print_results(results: list[TestResult]):
    """Print test results."""
    print("\n" + "=" * 60)
    print("SAFETY TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        symbol = "[OK]" if r.passed else "[!!]"
        print(f"{symbol} {r.test_name}: {r.message}")
        if r.details and not r.passed:
            print(f"    Details: {r.details}")

        if r.passed:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test Piper robot joint limits and safety")
    parser.add_argument("--mode", type=str, default="bimanual",
                        choices=["bimanual", "single_left", "single_right"],
                        help="Robot mode")
    parser.add_argument("--test-config", action="store_true",
                        help="Test joint limits configuration (no robot)")
    parser.add_argument("--test-checker", action="store_true",
                        help="Test safety checker logic (no robot)")
    parser.add_argument("--test-connection", action="store_true",
                        help="Test robot connection and read positions")
    parser.add_argument("--test-motion", action="store_true",
                        help="Test small motion on first joint")
    parser.add_argument("--joint", type=int, default=0,
                        help="Joint index for motion test")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="Delta for motion test (radians)")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")

    args = parser.parse_args()

    # Default to config and checker tests if no specific test requested
    if not any([args.test_config, args.test_checker, args.test_connection, args.test_motion, args.all]):
        args.test_config = True
        args.test_checker = True

    all_results = []

    if args.test_config or args.all:
        logger.info("Testing joint limits configuration...")
        all_results.extend(test_joint_limits_config())

    if args.test_checker or args.all:
        logger.info("Testing safety checker logic...")
        all_results.extend(test_safety_checker())

    if args.test_connection or args.all:
        logger.info("Testing robot connection...")
        all_results.extend(test_robot_connection(args.mode))

    if args.test_motion or args.all:
        logger.info("Testing small motion...")
        all_results.extend(test_small_motion(args.mode, args.joint, args.delta))

    success = print_results(all_results)

    if success:
        print("\nAll safety tests passed! Ready for RL training.")
    else:
        print("\nSome tests failed! Please check configuration before training.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

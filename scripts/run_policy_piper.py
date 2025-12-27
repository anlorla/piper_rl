#!/usr/bin/env python3
"""
Deploy a trained policy to the Piper dual-arm robot.

Usage:
    # Make sure piper_ros driver nodes are running first, then:
    python scripts/run_policy_piper.py \
        --policy-host 127.0.0.1 \
        --policy-port 8000 \
        --prompt "Push the block to the right"

    # Or use a local checkpoint:
    python scripts/run_policy_piper.py \
        --checkpoint outputs/train/act_piper/checkpoints/last/pretrained_model \
        --prompt "Push the block"
"""

import argparse
import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_with_websocket_policy(args):
    """Run with remote policy server (like openpi)."""
    from openpi_client import websocket_client_policy, image_tools
    from lerobot.robots.piper_ros import PiperRos, PiperRosConfig

    # Create robot
    config = PiperRosConfig(
        mode="bimanual",
        image_size=(256, 256),
        control_freq=args.freq,
        max_delta_pos=args.max_delta,
    )
    robot = PiperRos(config)
    robot.connect()

    # Create policy client
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )

    # Action buffer for sliding window
    action_buffer = None
    action_index = 0
    replan_threshold = 4

    rate_sleep = 1.0 / args.freq
    logger.info(f"Running at {args.freq} Hz, prompt: {args.prompt}")

    try:
        while True:
            start_time = time.time()

            # Get observation
            obs = robot.get_observation()

            # Check if need to re-predict
            if action_buffer is None or action_index >= len(action_buffer) - replan_threshold:
                # Prepare observation for policy
                policy_obs = {
                    "observation/image": obs["cam_top"],
                    "observation/wrist_image": obs["cam_left_wrist"],
                    "observation/right_wrist_image": obs["cam_right_wrist"],
                    "observation/state": _obs_to_state(obs),
                    "prompt": args.prompt,
                }

                logger.info("Requesting new action chunk...")
                result = client.infer(policy_obs)
                action_buffer = np.array(result["actions"])
                action_index = 0
                logger.info(f"Received {len(action_buffer)} actions")

            # Get current action
            action = action_buffer[action_index]
            action_index += 1

            # Convert to robot action format
            robot_action = _action_to_robot(action)

            # Send to robot
            robot.send_action(robot_action)

            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            if elapsed < rate_sleep:
                time.sleep(rate_sleep - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.disconnect()


def run_with_local_policy(args):
    """Run with local LeRobot checkpoint."""
    import torch
    from lerobot.common.policies.factory import make_policy
    from lerobot.robots.piper_ros import PiperRos, PiperRosConfig

    # Create robot
    config = PiperRosConfig(
        mode="bimanual",
        image_size=(224, 224),  # ACT typically uses 224
        control_freq=args.freq,
        max_delta_pos=args.max_delta,
    )
    robot = PiperRos(config)
    robot.connect()

    # Load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = make_policy(hydra_cfg_path=args.checkpoint)
    policy.to(device)
    policy.eval()

    rate_sleep = 1.0 / args.freq
    logger.info(f"Running at {args.freq} Hz with local policy")

    try:
        while True:
            start_time = time.time()

            # Get observation
            obs = robot.get_observation()

            # Convert to policy input format
            policy_obs = _obs_to_policy_input(obs, device)

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(policy_obs)

            # Convert to numpy and send to robot
            action_np = action.cpu().numpy()
            robot_action = _action_to_robot(action_np)
            robot.send_action(robot_action)

            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            if elapsed < rate_sleep:
                time.sleep(rate_sleep - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.disconnect()


def _obs_to_state(obs: dict) -> np.ndarray:
    """Convert robot observation to state vector."""
    left_joints = []
    right_joints = []
    for i in range(6):
        left_joints.append(obs.get(f"left_joint_{i+1}.pos", 0.0))
        right_joints.append(obs.get(f"right_joint_{i+1}.pos", 0.0))
    left_joints.append(obs.get("left_gripper.pos", 0.0))
    right_joints.append(obs.get("right_gripper.pos", 0.0))
    return np.array(left_joints + right_joints, dtype=np.float32)


def _action_to_robot(action: np.ndarray) -> dict:
    """Convert action array to robot action dict."""
    # Assuming action is 14-dim: [left_7, right_7]
    robot_action = {}
    for i in range(6):
        robot_action[f"left_joint_{i+1}.pos"] = float(action[i])
        robot_action[f"right_joint_{i+1}.pos"] = float(action[7 + i])
    robot_action["left_gripper.pos"] = float(action[6])
    robot_action["right_gripper.pos"] = float(action[13])
    return robot_action


def _obs_to_policy_input(obs: dict, device) -> dict:
    """Convert observation to policy input tensor dict."""
    import torch

    # Images: [B, C, H, W]
    images = {}
    for key in ["cam_top", "cam_left_wrist", "cam_right_wrist"]:
        if key in obs:
            img = obs[key]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            images[key] = img.to(device)

    # State
    state = _obs_to_state(obs)
    state = torch.from_numpy(state).unsqueeze(0).to(device)

    return {
        "observation.images.cam_top": images.get("cam_top"),
        "observation.images.cam_left_wrist": images.get("cam_left_wrist"),
        "observation.images.cam_right_wrist": images.get("cam_right_wrist"),
        "observation.state": state,
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy policy to Piper robot")
    parser.add_argument("--freq", type=int, default=10, help="Control frequency (Hz)")
    parser.add_argument("--max-delta", type=float, default=0.1, help="Max joint position change per step (rad)")

    # Policy source (choose one)
    parser.add_argument("--policy-host", type=str, help="Remote policy server host")
    parser.add_argument("--policy-port", type=int, default=8000, help="Remote policy server port")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path")

    parser.add_argument("--prompt", type=str, default="", help="Task prompt for VLA")

    args = parser.parse_args()

    if args.checkpoint:
        run_with_local_policy(args)
    elif args.policy_host:
        run_with_websocket_policy(args)
    else:
        parser.error("Either --checkpoint or --policy-host must be specified")


if __name__ == "__main__":
    main()
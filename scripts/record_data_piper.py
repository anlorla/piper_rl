#!/usr/bin/env python3
"""
Record teleoperation data from Piper dual-arm robot.

This script records data in LeRobot format for training policies.

Usage:
    # Record 50 episodes:
    python scripts/record_data_piper.py \
        --output-dir data/piper_push_task \
        --num-episodes 50 \
        --task "push block to the right"

Architecture:
    Leader arms (teleoperation) -> ROS -> Follower arms
                                    |
                                    v
                            This script records
                            observations + actions
"""

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def record_episode(robot, episode_idx: int, task: str, freq: int = 10) -> dict:
    """
    Record a single episode.

    Returns:
        dict with keys: observations, actions, timestamps
    """
    observations = []
    actions = []
    timestamps = []

    rate_sleep = 1.0 / freq
    logger.info(f"Episode {episode_idx}: Press ENTER to start recording, 'q' + ENTER to finish")
    input()

    logger.info("Recording... Press 'q' + ENTER to stop")
    start_time = time.time()

    # Start recording in a separate thread for input handling
    import threading
    stop_flag = threading.Event()

    def wait_for_stop():
        while not stop_flag.is_set():
            try:
                if input() == 'q':
                    stop_flag.set()
                    break
            except EOFError:
                break

    input_thread = threading.Thread(target=wait_for_stop, daemon=True)
    input_thread.start()

    frame_idx = 0
    while not stop_flag.is_set():
        frame_start = time.time()

        # Get current observation (this is also the action for teleoperation)
        obs = robot.get_observation()

        # Store observation
        observations.append(obs)

        # For teleoperation, action = current joint positions (the leader arm positions)
        # In a typical setup, the leader arms send commands to follower arms
        # Here we record what the follower arms are doing (which follows leader)
        action = {k: v for k, v in obs.items() if k.endswith('.pos')}
        actions.append(action)

        timestamps.append(time.time() - start_time)
        frame_idx += 1

        # Maintain frequency
        elapsed = time.time() - frame_start
        if elapsed < rate_sleep:
            time.sleep(rate_sleep - elapsed)

    duration = time.time() - start_time
    logger.info(f"Episode {episode_idx}: Recorded {frame_idx} frames in {duration:.1f}s ({frame_idx/duration:.1f} Hz)")

    return {
        "observations": observations,
        "actions": actions,
        "timestamps": timestamps,
        "task": task,
        "episode_idx": episode_idx,
    }


def save_episode_lerobot_format(episode_data: dict, output_dir: Path, fps: int = 10):
    """
    Save episode in LeRobot dataset format.

    LeRobot expects:
    - data/chunk-XXX/episode_YYYYYY.parquet (tabular data)
    - videos/chunk-XXX/observation.images.XXX/episode_YYYYYY.mp4
    """
    import pandas as pd

    episode_idx = episode_data["episode_idx"]
    observations = episode_data["observations"]
    actions = episode_data["actions"]
    timestamps = episode_data["timestamps"]

    output_dir = Path(output_dir)
    chunk_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Build dataframe rows
    rows = []
    for i, (obs, action, ts) in enumerate(zip(observations, actions, timestamps)):
        row = {
            "episode_index": episode_idx,
            "frame_index": i,
            "timestamp": ts,
            "task": episode_data["task"],
        }

        # Add observation state (joint positions)
        state = []
        for arm in ["left", "right"]:
            for j in range(6):
                key = f"{arm}_joint_{j+1}.pos"
                state.append(obs.get(key, 0.0))
            state.append(obs.get(f"{arm}_gripper.pos", 0.0))
        row["observation.state"] = state

        # Add action
        action_vec = []
        for arm in ["left", "right"]:
            for j in range(6):
                key = f"{arm}_joint_{j+1}.pos"
                action_vec.append(action.get(key, 0.0))
            action_vec.append(action.get(f"{arm}_gripper.pos", 0.0))
        row["action"] = action_vec

        rows.append(row)

    # Save parquet
    df = pd.DataFrame(rows)
    parquet_path = chunk_dir / f"episode_{episode_idx:06d}.parquet"
    df.to_parquet(parquet_path)
    logger.info(f"Saved {parquet_path}")

    # Save videos
    for cam_name in ["cam_top", "cam_left_wrist", "cam_right_wrist"]:
        cam_video_dir = video_dir / f"observation.images.{cam_name}"
        cam_video_dir.mkdir(parents=True, exist_ok=True)
        video_path = cam_video_dir / f"episode_{episode_idx:06d}.mp4"

        # Get image size from first observation
        if observations and cam_name in observations[0]:
            first_img = observations[0][cam_name]
            h, w = first_img.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

            for obs in observations:
                if cam_name in obs:
                    img = obs[cam_name]
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    writer.write(img_bgr)

            writer.release()
            logger.info(f"Saved {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Record teleoperation data from Piper robot")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--task", type=str, default="manipulation task", help="Task description")
    parser.add_argument("--freq", type=int, default=10, help="Recording frequency (Hz)")

    args = parser.parse_args()

    from lerobot.robots.piper_ros import PiperRos, PiperRosConfig

    # Create robot
    config = PiperRosConfig(
        mode="bimanual",
        image_size=(224, 224),
        control_freq=args.freq,
    )
    robot = PiperRos(config)

    try:
        robot.connect()
        logger.info("Robot connected. Ready to record.")

        for ep_idx in range(args.num_episodes):
            episode_data = record_episode(robot, ep_idx, args.task, args.freq)
            save_episode_lerobot_format(episode_data, args.output_dir, args.freq)
            logger.info(f"Completed episode {ep_idx + 1}/{args.num_episodes}")

        logger.info(f"All episodes recorded to {args.output_dir}")

    except KeyboardInterrupt:
        logger.info("Recording interrupted")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
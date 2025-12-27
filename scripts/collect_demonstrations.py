#!/usr/bin/env python3
"""
Collect demonstration data from Piper dual-arm robot for RL training.

This script collects human demonstrations via teleoperation and saves them
in LeRobot format for training policies or reward classifiers.

Usage:
    # Step 1: Start ROS driver
    # Step 2: Run this script

    # Collect demonstrations for a task:
    python scripts/collect_demonstrations.py \
        --repo-id zeno/piper_pick_cube \
        --task "pick up the red cube" \
        --num-episodes 20 \
        --fps 10

    # With success labeling (for reward classifier training):
    python scripts/collect_demonstrations.py \
        --repo-id zeno/piper_pick_cube \
        --task "pick up the red cube" \
        --num-episodes 20 \
        --label-success

Controls:
    - Use Leader arms to teleoperate the robot
    - Press ENTER to start recording an episode
    - Press 's' + ENTER to mark episode as SUCCESS and save
    - Press 'f' + ENTER to mark episode as FAILURE and save
    - Press 'r' + ENTER to REDO current episode (discard)
    - Press 'q' + ENTER to QUIT recording
    - Press Ctrl+C to force quit
"""

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    """Data for a single episode."""
    episode_idx: int
    task: str
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    success: bool | None = None  # None = not labeled


class KeyboardController:
    """Non-blocking keyboard input for episode control."""

    def __init__(self):
        self.command = None
        self._stop = False
        self._thread = None

    def start(self):
        """Start listening for keyboard input."""
        self._stop = False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening."""
        self._stop = True

    def _listen(self):
        while not self._stop:
            try:
                cmd = input()
                self.command = cmd.lower().strip()
            except EOFError:
                break

    def get_command(self) -> str | None:
        """Get and clear current command."""
        cmd = self.command
        self.command = None
        return cmd


def record_episode(
    robot,
    episode_idx: int,
    task: str,
    fps: int = 10,
    label_success: bool = False,
    display_cameras: bool = True,
) -> EpisodeData | None:
    """
    Record a single episode of teleoperation.

    Returns:
        EpisodeData if episode should be saved, None if discarded
    """
    episode = EpisodeData(episode_idx=episode_idx, task=task)
    rate_sleep = 1.0 / fps

    # Instructions
    print("\n" + "=" * 60)
    print(f"Episode {episode_idx}")
    print("=" * 60)
    print("Controls:")
    print("  ENTER     - Start recording")
    print("  's'+ENTER - Stop and mark SUCCESS")
    print("  'f'+ENTER - Stop and mark FAILURE")
    print("  'r'+ENTER - Redo episode (discard)")
    print("  'q'+ENTER - Quit")
    print("-" * 60)

    # Wait for start
    while True:
        cmd = input("Press ENTER to start recording: ")
        if cmd == '':
            break
        elif cmd.lower() == 'q':
            return None

    print("Recording... (use commands above to stop)")

    # Start keyboard listener
    kb = KeyboardController()
    kb.start()

    start_time = time.time()
    frame_idx = 0
    running = True

    while running:
        frame_start = time.time()

        # Get observation
        obs = robot.get_observation()

        # Store observation
        episode.observations.append(obs)

        # For teleoperation, action = current joint positions
        action = {k: v for k, v in obs.items() if k.endswith('.pos')}
        episode.actions.append(action)

        # Timestamp
        episode.timestamps.append(time.time() - start_time)

        # Display cameras
        if display_cameras:
            for cam_key in ["cam_top", "cam_left_wrist", "cam_right_wrist"]:
                if cam_key in obs:
                    img_bgr = cv2.cvtColor(obs[cam_key], cv2.COLOR_RGB2BGR)
                    cv2.imshow(cam_key, img_bgr)
            cv2.waitKey(1)

        frame_idx += 1

        # Check for commands
        cmd = kb.get_command()
        if cmd is not None:
            if cmd == 's':
                episode.success = True
                running = False
                logger.info("Episode marked as SUCCESS")
            elif cmd == 'f':
                episode.success = False
                running = False
                logger.info("Episode marked as FAILURE")
            elif cmd == 'r':
                logger.info("Episode DISCARDED, redo")
                kb.stop()
                return record_episode(robot, episode_idx, task, fps, label_success, display_cameras)
            elif cmd == 'q':
                kb.stop()
                return None

        # Maintain frequency
        elapsed = time.time() - frame_start
        if elapsed < rate_sleep:
            time.sleep(rate_sleep - elapsed)

    kb.stop()
    duration = time.time() - start_time
    actual_fps = frame_idx / duration if duration > 0 else 0

    print(f"Recorded {frame_idx} frames in {duration:.1f}s ({actual_fps:.1f} Hz)")

    # If success labeling is required but not provided
    if label_success and episode.success is None:
        while True:
            label = input("Was this episode successful? (s=success, f=failure): ").lower()
            if label == 's':
                episode.success = True
                break
            elif label == 'f':
                episode.success = False
                break

    return episode


def save_episode_to_lerobot(
    episode: EpisodeData,
    dataset,
    fps: int,
) -> None:
    """Save episode to LeRobot dataset."""
    observations = episode.observations
    actions = episode.actions

    for i, (obs, action) in enumerate(zip(observations, actions)):
        # Build observation state vector
        state = []
        for arm in ["left", "right"]:
            for j in range(1, 7):  # joints 1-6
                key = f"{arm}_joint_{j}.pos"
                state.append(float(obs.get(key, 0.0)))
            # gripper
            state.append(float(obs.get(f"{arm}_gripper.pos", 0.0)))

        # Build action vector (same as state for teleoperation)
        action_vec = []
        for arm in ["left", "right"]:
            for j in range(1, 7):
                key = f"{arm}_joint_{j}.pos"
                action_vec.append(float(action.get(key, 0.0)))
            action_vec.append(float(action.get(f"{arm}_gripper.pos", 0.0)))

        # Build frame
        frame = {
            "observation.state": np.array(state, dtype=np.float32),
            "action": np.array(action_vec, dtype=np.float32),
            "task": episode.task,
        }

        # Add success label if available (for reward classifier training)
        if episode.success is not None:
            # Only the last frame gets success=True, others get False
            is_last_frame = (i == len(observations) - 1)
            frame["success"] = episode.success and is_last_frame

        # Add images
        for cam_key in ["cam_top", "cam_left_wrist", "cam_right_wrist"]:
            if cam_key in obs:
                frame[f"observation.images.{cam_key}"] = obs[cam_key]

        dataset.add_frame(frame)

    dataset.save_episode()
    logger.info(f"Saved episode {episode.episode_idx} ({len(observations)} frames)")


def create_dataset(repo_id: str, fps: int, root: str | None = None):
    """Create LeRobot dataset with proper features."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Define features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),  # 7 joints x 2 arms
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": None,
        },
        "success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "observation.images.cam_top": {
            "dtype": "video",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id,
        fps,
        root=root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Collect demonstration data from Piper robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="Dataset repository ID (e.g., 'zeno/piper_pick_cube')"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task description"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Recording frequency (Hz)"
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Root directory for dataset (default: ~/.cache/huggingface/lerobot)"
    )
    parser.add_argument(
        "--label-success", action="store_true",
        help="Label each episode as success/failure (for reward classifier)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable camera display"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push dataset to HuggingFace Hub when done"
    )

    args = parser.parse_args()

    # Import robot
    from lerobot.robots.piper_ros import PiperRos, PiperRosConfig

    # Create robot
    config = PiperRosConfig(
        mode="bimanual",
        image_size=(224, 224),
        control_freq=args.fps,
    )
    robot = PiperRos(config)

    # Create dataset
    logger.info(f"Creating dataset: {args.repo_id}")
    dataset = create_dataset(args.repo_id, args.fps, args.root)

    try:
        robot.connect()
        logger.info("Robot connected. Ready to record.")
        logger.info(f"Task: {args.task}")
        logger.info(f"Target: {args.num_episodes} episodes")

        episode_idx = 0
        while episode_idx < args.num_episodes:
            episode = record_episode(
                robot=robot,
                episode_idx=episode_idx,
                task=args.task,
                fps=args.fps,
                label_success=args.label_success,
                display_cameras=not args.no_display,
            )

            if episode is None:
                logger.info("Recording stopped by user")
                break

            # Save episode
            save_episode_to_lerobot(episode, dataset, args.fps)
            episode_idx += 1

            print(f"\nProgress: {episode_idx}/{args.num_episodes} episodes\n")

        # Summary
        print("\n" + "=" * 60)
        print("RECORDING COMPLETE")
        print("=" * 60)
        print(f"Total episodes: {episode_idx}")
        print(f"Dataset: {args.repo_id}")

        if args.push_to_hub and episode_idx > 0:
            logger.info("Pushing dataset to HuggingFace Hub...")
            dataset.push_to_hub()
            logger.info("Dataset pushed successfully!")

    except KeyboardInterrupt:
        logger.info("\nRecording interrupted by user")
    finally:
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
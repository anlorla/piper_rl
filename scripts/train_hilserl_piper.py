#!/usr/bin/env python3
"""
HIL-SERL Training for Piper Dual-Arm Robot.

Human-in-the-Loop Sample Efficient Reinforcement Learning.

Usage:
    # Step 1: Make sure piper_ros driver is running
    # Step 2: Run training

    # Option A: Use reward classifier (automatic)
    python scripts/train_hilserl_piper.py \
        --task "pick up the cube" \
        --reward-classifier-path models/reward_classifier.pt \
        --output-dir outputs/hilserl_piper

    # Option B: Use keyboard for human reward (manual)
    # Default reward = 0, press SPACE = 1 (success)
    python scripts/train_hilserl_piper.py \
        --task "pick up the cube" \
        --keyboard-reward \
        --output-dir outputs/hilserl_piper

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    HIL-SERL Pipeline                     │
    ├─────────────────────────────────────────────────────────┤
    │   ┌──────────────┐          ┌──────────────┐            │
    │   │  Actor进程   │ ←──────→ │  Learner进程  │            │
    │   │  (数据采集)  │ transitions│  (SAC训练)  │            │
    │   │             │ ←──────→  │             │            │
    │   └──────┬──────┘ parameters└──────────────┘            │
    │          │                                              │
    │          ↓                                              │
    │   ┌──────────────┐    ┌──────────────┐                 │
    │   │ PiperRosEnv  │←───│ Teleoperator │                 │
    │   │  (双臂机器人) │    │ (手柄/Leader) │                 │
    │   └──────────────┘    └──────────────┘                 │
    └─────────────────────────────────────────────────────────┘
"""

import argparse
import logging
import multiprocessing as mp
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path
from queue import Empty, Full

import numpy as np
import torch
import torch.optim as optim


class KeyboardRewardInput:
    """Non-blocking keyboard input for human reward labeling."""

    def __init__(self, success_key: str = " "):
        """
        Args:
            success_key: Key to press for success reward (default: space bar)
        """
        self.success_key = success_key
        self.old_settings = None
        self._enabled = False

    def enable(self):
        """Enable raw terminal mode for non-blocking input."""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._enabled = True
        except Exception as e:
            logging.warning(f"Could not enable keyboard input: {e}")
            self._enabled = False

    def disable(self):
        """Restore terminal settings."""
        if self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass
        self._enabled = False

    def get_reward(self, default: float = 0.0, success_reward: float = 1.0) -> float:
        """
        Check for keyboard input and return reward.

        Args:
            default: Default reward when no key pressed
            success_reward: Reward when success key is pressed

        Returns:
            Reward value
        """
        if not self._enabled:
            return default

        try:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                if key == self.success_key:
                    return success_reward
        except Exception:
            pass
        return default

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Training hyperparameters
LOG_EVERY = 10
SEND_EVERY = 10
DEFAULT_MAX_EPISODES = 100
DEFAULT_MAX_STEPS = 300


def make_policy_obs(obs: dict, device: str = "cpu") -> dict:
    """Convert environment observation to policy input format."""
    policy_obs = {
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device),
    }

    if "pixels" in obs:
        for key, img in obs["pixels"].items():
            # Convert HWC to CHW and normalize
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            policy_obs[f"observation.images.{key}"] = img_tensor.to(device)

    return policy_obs


def run_learner(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_state_dict: dict,
    policy_config: dict,
    online_buffer_capacity: int = 10000,
    offline_buffer_capacity: int = 5000,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Learner process - trains SAC policy on collected transitions.
    """
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.policies.sac.modeling_sac import SACPolicy
    from lerobot.rl.buffer import ReplayBuffer

    # Create policy
    policy_cfg = SACConfig(**policy_config)
    policy = SACPolicy(policy_cfg)
    policy.load_state_dict(policy_state_dict)
    policy.train()
    policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Create replay buffers
    online_buffer = ReplayBuffer(capacity=online_buffer_capacity, device=device)
    offline_buffer = ReplayBuffer(capacity=offline_buffer_capacity, device=device)

    logger.info(f"[LEARNER] Started on device {device}")
    logger.info(f"[LEARNER] Online buffer capacity: {online_buffer_capacity}")
    logger.info(f"[LEARNER] Offline buffer capacity: {offline_buffer_capacity}")

    training_step = 0

    while not shutdown_event.is_set():
        # Get transitions from actor
        try:
            transitions = transitions_queue.get(timeout=0.1)
            for transition in transitions:
                # Add all transitions to online buffer
                online_buffer.add(**transition)

                # Add only intervention transitions to offline buffer
                is_intervention = transition.get("complementary_info", {}).get("is_intervention", False)
                if is_intervention:
                    offline_buffer.add(**transition)
                    logger.info(f"[LEARNER] Intervention detected! Offline buffer: {len(offline_buffer)}")

        except Empty:
            pass

        # Train if enough data
        min_samples = policy_cfg.online_step_before_learning
        if len(online_buffer) >= min_samples and len(offline_buffer) >= batch_size // 2:
            # HIL-SERL: Mix online and offline data
            online_batch = online_buffer.sample(batch_size // 2)
            offline_batch = offline_buffer.sample(batch_size // 2)

            # Combine batches
            batch = {}
            for key in online_batch:
                if key in offline_batch:
                    batch[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
                else:
                    batch[key] = online_batch[key]

            # Training step
            loss, _ = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % LOG_EVERY == 0:
                logger.info(
                    f"[LEARNER] Step {training_step}, Loss: {loss.item():.4f}, "
                    f"Online: {len(online_buffer)}, Offline: {len(offline_buffer)}"
                )

            # Send updated parameters to actor
            if training_step % SEND_EVERY == 0:
                try:
                    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
                    parameters_queue.put_nowait(state_dict)
                except Full:
                    pass

    logger.info("[LEARNER] Finished")


def run_actor(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_state_dict: dict,
    policy_config: dict,
    reward_classifier_path: str | None,
    task: str,
    max_episodes: int,
    max_steps: int,
    device: str = "cpu",
    output_dir: Path | None = None,
    teleop_type: str = "leader",
    use_keyboard_reward: bool = False,
    keyboard_success_key: str = " ",
):
    """
    Actor process - interacts with environment and collects data.

    Args:
        teleop_type: Type of teleoperator for intervention
            - "leader": Piper leader arms (recommended)
            - "gamepad": Gamepad controller
            - "none": No intervention
        use_keyboard_reward: If True, use keyboard input for reward (default=0, press key=1)
        keyboard_success_key: Key to press for success reward (default: space bar)
    """
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.policies.sac.modeling_sac import SACPolicy
    from lerobot.rl.piper_env import make_piper_env

    # Create policy
    policy_cfg = SACConfig(**policy_config)
    policy = SACPolicy(policy_cfg)
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    policy.to(device)

    # Create environment
    env = make_piper_env(
        mode="bimanual",
        use_gripper=True,
        fps=10,
    )

    # Create teleoperator for human intervention
    teleop_device = None
    if teleop_type == "leader":
        try:
            from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig
            teleop_config = PiperLeaderConfig(
                mode="bimanual",
                intervention_threshold=0.02,  # radians
            )
            teleop_device = PiperLeader(teleop_config)
            teleop_device.connect()
            logger.info("[ACTOR] Piper Leader arms connected for intervention")
        except Exception as e:
            logger.warning(f"[ACTOR] Could not connect leader arms: {e}")
    elif teleop_type == "gamepad":
        try:
            from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
            from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
            teleop_config = GamepadTeleopConfig(use_gripper=True)
            teleop_device = GamepadTeleop(teleop_config)
            teleop_device.connect()
            logger.info("[ACTOR] Gamepad connected for intervention")
        except Exception as e:
            logger.warning(f"[ACTOR] Could not connect gamepad: {e}")

    # Load reward classifier if provided
    reward_classifier = None
    if reward_classifier_path:
        try:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
            reward_classifier = Classifier.from_pretrained(reward_classifier_path)
            reward_classifier.to(device)
            reward_classifier.eval()
            logger.info("[ACTOR] Reward classifier loaded")
        except Exception as e:
            logger.warning(f"[ACTOR] Could not load reward classifier: {e}")

    logger.info(f"[ACTOR] Started. Task: {task}")
    logger.info(f"[ACTOR] Max episodes: {max_episodes}, Max steps: {max_steps}")

    # Setup keyboard reward input if enabled
    keyboard_reward = None
    if use_keyboard_reward:
        keyboard_reward = KeyboardRewardInput(success_key=keyboard_success_key)
        keyboard_reward.enable()
        logger.info(f"[ACTOR] Keyboard reward enabled. Press '{keyboard_success_key}' (space) for success (reward=1), default reward=0")

    try:
        for episode in range(max_episodes):
            if shutdown_event.is_set():
                break

            obs, _ = env.reset()
            episode_reward = 0.0
            episode_successes = 0  # Count keyboard success presses
            step = 0
            episode_transitions = []

            logger.info(f"[ACTOR] Episode {episode + 1}/{max_episodes}")

            while step < max_steps and not shutdown_event.is_set():
                # Check for parameter updates
                try:
                    new_params = parameters_queue.get_nowait()
                    policy.load_state_dict(new_params)
                    logger.info("[ACTOR] Updated policy from learner")
                except Empty:
                    pass

                # Get action from policy
                policy_obs = make_policy_obs(obs, device=device)
                with torch.no_grad():
                    action_tensor = policy.select_action(policy_obs)
                action = action_tensor.squeeze(0).cpu().numpy()

                # Check for intervention
                is_intervention = False
                if teleop_device:
                    teleop_events = teleop_device.get_teleop_events()
                    is_intervention = teleop_events.get("is_intervention", False)

                    if is_intervention:
                        # Use teleop action instead
                        teleop_action = teleop_device.get_action()
                        if "positions" in teleop_action:
                            action = teleop_action["positions"]

                # Step environment
                next_obs, env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Compute reward (priority: keyboard > classifier > env)
                reward = env_reward
                if keyboard_reward is not None:
                    # Keyboard reward: default=0, press key=1
                    reward = keyboard_reward.get_reward(default=0.0, success_reward=1.0)
                    if reward > 0:
                        episode_successes += 1
                        logger.info(f"[ACTOR] Step {step}: SUCCESS! (reward=1)")
                elif reward_classifier is not None:
                    policy_next_obs = make_policy_obs(next_obs, device=device)
                    with torch.no_grad():
                        reward = reward_classifier.predict_reward(policy_next_obs)
                        if isinstance(reward, torch.Tensor):
                            reward = reward.item()

                # Store transition
                transition = {
                    "state": policy_obs,
                    "action": action,
                    "reward": float(reward),
                    "next_state": make_policy_obs(next_obs, device=device),
                    "done": done,
                    "truncated": truncated,
                    "complementary_info": {
                        "is_intervention": is_intervention,
                    },
                }
                episode_transitions.append(transition)

                episode_reward += reward
                step += 1
                obs = next_obs

                if done:
                    break

            # Send transitions to learner
            try:
                transitions_queue.put_nowait(episode_transitions)
            except Full:
                logger.warning("[ACTOR] Transition queue full, dropping episode")

            if use_keyboard_reward:
                logger.info(f"[ACTOR] Episode {episode + 1} finished. Steps: {step}, Reward: {episode_reward:.2f}, Successes: {episode_successes}")
            else:
                logger.info(f"[ACTOR] Episode {episode + 1} finished. Steps: {step}, Reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        logger.info("[ACTOR] Interrupted")
    finally:
        env.close()
        if teleop_device:
            teleop_device.disconnect()
        if keyboard_reward is not None:
            keyboard_reward.disable()
        if output_dir:
            policy.save_pretrained(output_dir / "final_policy")
            logger.info(f"[ACTOR] Policy saved to {output_dir / 'final_policy'}")

    logger.info("[ACTOR] Finished")


def main():
    parser = argparse.ArgumentParser(description="HIL-SERL training for Piper robot")

    # Task configuration
    parser.add_argument("--task", type=str, required=True, help="Task description")
    parser.add_argument("--reward-classifier-path", type=str, help="Path to reward classifier")

    # Training parameters
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)

    # Device configuration
    parser.add_argument("--learner-device", type=str, default="cuda")
    parser.add_argument("--actor-device", type=str, default="cpu")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/hilserl_piper")

    # Teleop (Human Intervention)
    parser.add_argument(
        "--teleop",
        type=str,
        default="leader",
        choices=["leader", "gamepad", "none"],
        help="Teleoperator type for human intervention: leader (piper leader arms), gamepad, or none"
    )
    parser.add_argument("--intervention-threshold", type=float, default=0.02,
                        help="Joint movement threshold for detecting intervention (radians)")

    # Human Reward Input
    parser.add_argument("--keyboard-reward", action="store_true",
                        help="Enable keyboard input for reward (default=0, press space=1)")
    parser.add_argument("--success-key", type=str, default=" ",
                        help="Key to press for success reward (default: space bar)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define policy configuration
    # These should match your robot's observation/action space
    policy_config = {
        "device": args.actor_device,
        "input_features": {
            "observation.state": {"shape": (14,), "dtype": "float32"},  # 7 joints x 2 arms
            "observation.images.cam_top": {"shape": (3, 224, 224), "dtype": "float32"},
            "observation.images.cam_left_wrist": {"shape": (3, 224, 224), "dtype": "float32"},
            "observation.images.cam_right_wrist": {"shape": (3, 224, 224), "dtype": "float32"},
        },
        "output_features": {
            "action": {"shape": (14,), "dtype": "float32"},  # 7 joints x 2 arms
        },
        "online_step_before_learning": 100,
    }

    # Create initial policy
    from lerobot.policies.sac.configuration_sac import SACConfig
    from lerobot.policies.sac.modeling_sac import SACPolicy

    initial_policy = SACPolicy(SACConfig(**policy_config))
    initial_state_dict = {k: v.cpu() for k, v in initial_policy.state_dict().items()}

    # Create communication queues
    transitions_queue = mp.Queue(maxsize=10)
    parameters_queue = mp.Queue(maxsize=2)
    shutdown_event = mp.Event()

    # Signal handler
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create processes
    learner_process = mp.Process(
        target=run_learner,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            initial_state_dict,
            policy_config,
        ),
        kwargs={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "device": args.learner_device,
        },
    )

    actor_process = mp.Process(
        target=run_actor,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            initial_state_dict,
            policy_config,
            args.reward_classifier_path,
            args.task,
            args.max_episodes,
            args.max_steps,
        ),
        kwargs={
            "device": args.actor_device,
            "output_dir": output_dir,
            "teleop_type": args.teleop,
            "use_keyboard_reward": args.keyboard_reward,
            "keyboard_success_key": args.success_key,
        },
    )

    logger.info("Starting HIL-SERL training for Piper robot")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output directory: {output_dir}")

    learner_process.start()
    actor_process.start()

    try:
        actor_process.join()
        shutdown_event.set()
        learner_process.join(timeout=10)
    except KeyboardInterrupt:
        logger.info("Main process interrupted")
        shutdown_event.set()
        actor_process.join(timeout=5)
        learner_process.join(timeout=10)
    finally:
        if learner_process.is_alive():
            learner_process.terminate()
        if actor_process.is_alive():
            actor_process.terminate()

    logger.info("Training finished")


if __name__ == "__main__":
    main()

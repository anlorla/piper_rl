# HIL-SERL Training for Piper Robot

Human-in-the-Loop Sample Efficient Reinforcement Learning (HIL-SERL) on Piper dual-arm robot.

## Overview

HIL-SERL is an online reinforcement learning method that combines:
- **SAC (Soft Actor-Critic)**: Off-policy RL algorithm
- **Human Intervention**: Expert demonstrations during training
- **Dual Buffer**: Online buffer (all data) + Offline buffer (intervention data, higher priority)

```
┌─────────────────────────────────────────────────────────────┐
│                    HIL-SERL Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│   ┌──────────────┐          ┌──────────────┐                │
│   │  Actor进程   │ ←──────→ │  Learner进程  │                │
│   │  (数据采集)  │ transitions│  (SAC训练)  │                │
│   │             │ ←──────→  │             │                │
│   └──────┬──────┘ parameters└──────────────┘                │
│          │                                                  │
│          ↓                                                  │
│   ┌──────────────┐    ┌──────────────┐                     │
│   │ PiperRosEnv  │←───│ Teleoperator │                     │
│   │  (双臂机器人) │    │ (Leader臂)   │                     │
│   └──────────────┘    └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Hardware Setup
- Piper dual-arm robot (follower arms)
- Piper leader arms (for human intervention)
- RealSense cameras (top + wrist cameras)
- Ubuntu 20.04 with ROS Noetic

### 2. Software Dependencies
```bash
# Install ROS dependencies
sudo apt install ros-noetic-cv-bridge ros-noetic-sensor-msgs

# Install Python dependencies
pip install gymnasium numpy torch opencv-python

# Install LeRobot
cd /home/zeno-yifan/NPM-Project/PiPER-RL/lerobot
pip install -e .
```

### 3. Start ROS Nodes
```bash
# Terminal 1: Start piper_ros driver (from NPM-Ros)
cd /home/zeno-yifan/NPM-Project/NPM-Ros/piper_ros
source devel/setup.bash
roslaunch piper piper_dual_arm.launch

# Terminal 2: Start cameras
roslaunch realsense2_camera rs_camera.launch camera:=realsense_top
roslaunch realsense2_camera rs_camera.launch camera:=realsense_left
roslaunch realsense2_camera rs_camera.launch camera:=realsense_right
```

## Training

### Basic Training
```bash
cd /home/zeno-yifan/NPM-Project/PiPER-RL

python scripts/train_hilserl_piper.py \
    --task "pick up the cube" \
    --max-episodes 100 \
    --max-steps 300 \
    --output-dir outputs/hilserl_piper
```

### Training with Reward Classifier
```bash
python scripts/train_hilserl_piper.py \
    --task "pick up the cube" \
    --reward-classifier-path models/reward_classifier.pt \
    --max-episodes 100 \
    --output-dir outputs/hilserl_piper
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | required | Task description |
| `--reward-classifier-path` | None | Path to trained reward classifier |
| `--max-episodes` | 100 | Maximum training episodes |
| `--max-steps` | 300 | Maximum steps per episode |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 3e-4 | Learning rate |
| `--learner-device` | cuda | Device for learner (training) |
| `--actor-device` | cpu | Device for actor (inference) |
| `--no-gamepad` | False | Disable gamepad intervention |

## Human Intervention (核心机制)

### What is Human Intervention?

Human intervention is the key feature of HIL-SERL. During training:
1. The policy controls the robot most of the time
2. When the policy makes mistakes, **human expert takes over**
3. Human demonstrates the correct action using **leader arms**
4. Intervention data goes to the **offline buffer** (higher learning priority)

### How Intervention Works

```python
# Detection mechanism in PiperLeader teleoperator
def get_teleop_events(self):
    # Check if any joint moved more than threshold
    delta = np.abs(current_joints - previous_joints)
    is_intervention = np.any(delta > threshold)  # default: 0.02 rad

    return {
        TeleopEvents.IS_INTERVENTION: is_intervention,
        ...
    }
```

### Intervention Methods

#### Method 1: Leader Arms (Recommended)
Use Piper leader arms for natural teleoperation:

```python
# Configure leader arm teleoperator
from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

config = PiperLeaderConfig(
    mode="bimanual",
    left_leader_topic="/robot/leader_left/joint_states",
    right_leader_topic="/robot/leader_right/joint_states",
    intervention_threshold=0.02,  # radians
)
teleop = PiperLeader(config)
teleop.connect()
```

**How to intervene:**
1. Grab the leader arm handles
2. Move the leader arms to demonstrate the correct motion
3. The system automatically detects movement > threshold as intervention
4. Your demonstration replaces the policy's action

#### Method 2: Gamepad
Use a gamepad for intervention:

```python
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig

config = GamepadTeleopConfig(
    use_gripper=True,
)
teleop = GamepadTeleop(config)
teleop.connect()
```

**Button mapping:**
- Left stick: Control left arm
- Right stick: Control right arm
- Triggers: Gripper open/close
- Button to activate intervention mode

### Data Flow During Intervention

```
Normal operation:
  Policy action → env.step() → Online Buffer only

During intervention:
  Human action → env.step() → Online Buffer + Offline Buffer
                                    ↓
                             Higher priority in training
```

### Intervention Tips

1. **When to intervene:**
   - Robot is about to drop an object
   - Robot is moving in wrong direction
   - Robot is stuck or oscillating
   - Critical moments (grasping, placing)

2. **How to intervene effectively:**
   - React quickly when you see mistakes
   - Provide smooth, deliberate demonstrations
   - Complete the corrective action fully
   - Release control gradually

3. **Intervention rate:**
   - Start with ~20-30% intervention rate
   - Reduce as policy improves
   - Minimum 5% to maintain offline buffer

## ROS Topics

### Subscribed Topics (Observation)
| Topic | Type | Description |
|-------|------|-------------|
| `/robot/arm_left/joint_states_single` | `sensor_msgs/JointState` | Left follower arm joints |
| `/robot/arm_right/joint_states_single` | `sensor_msgs/JointState` | Right follower arm joints |
| `/robot/arm_left/end_pose` | `geometry_msgs/PoseStamped` | Left arm end-effector pose |
| `/robot/arm_right/end_pose` | `geometry_msgs/PoseStamped` | Right arm end-effector pose |
| `/realsense_*/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | Camera images |

### Published Topics (Action)
| Topic | Type | Description |
|-------|------|-------------|
| `/robot/arm_left/vla_joint_cmd` | `sensor_msgs/JointState` | Left arm joint commands |
| `/robot/arm_right/vla_joint_cmd` | `sensor_msgs/JointState` | Right arm joint commands |

### Leader Arm Topics (Intervention)
| Topic | Type | Description |
|-------|------|-------------|
| `/robot/leader_left/joint_states` | `sensor_msgs/JointState` | Left leader arm joints |
| `/robot/leader_right/joint_states` | `sensor_msgs/JointState` | Right leader arm joints |

## Code Structure

```
PiPER-RL/
├── lerobot/src/lerobot/
│   ├── robots/
│   │   └── piper_ros/
│   │       ├── piper_ros.py      # Robot class with ROS interface
│   │       └── config_piper_ros.py
│   ├── teleoperators/
│   │   └── piper_leader/
│   │       ├── piper_leader.py   # Leader arm teleoperator
│   │       └── config_piper_leader.py
│   └── rl/
│       ├── piper_env.py          # Gym environment
│       ├── actor.py              # Actor process
│       ├── learner.py            # Learner process
│       └── buffer.py             # Replay buffers
├── scripts/
│   └── train_hilserl_piper.py    # Training script
└── README_HIL_SERL.md            # This file
```

## Debugging

### Check ROS Topics
```bash
# List all topics
rostopic list | grep robot

# Check joint states
rostopic echo /robot/arm_left/joint_states_single

# Check leader arm
rostopic echo /robot/leader_left/joint_states

# Check camera
rostopic hz /realsense_top/color/image_raw/compressed
```

### Test Robot Connection
```python
from lerobot.robots.piper_ros import PiperRos, PiperRosConfig

config = PiperRosConfig(mode="bimanual")
robot = PiperRos(config)
robot.connect()

# Get observation
obs = robot.get_observation()
print("Joint positions:", obs)

# Get state vector
state = robot.get_state_vector()
print("State vector shape:", state.shape)

robot.disconnect()
```

### Test Teleoperator
```python
from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

config = PiperLeaderConfig(mode="bimanual")
teleop = PiperLeader(config)
teleop.connect()

import time
for _ in range(100):
    events = teleop.get_teleop_events()
    action = teleop.get_action()
    print(f"Intervention: {events['is_intervention']}, Action: {action['positions'][:3]}")
    time.sleep(0.1)

teleop.disconnect()
```

## FAQ

### Q: How does the system know when I'm intervening?
**A:** The system monitors the leader arm joint positions. When any joint moves more than the `intervention_threshold` (default 0.02 radians ≈ 1.15 degrees) between consecutive readings, it's detected as intervention.

### Q: What if I don't have leader arms?
**A:** You can use:
1. Gamepad with `--use-gamepad` flag
2. Keyboard with `--use-keyboard` flag (for end-effector control)
3. Train without intervention (slower learning)

### Q: How much intervention is needed?
**A:**
- Early training: 20-30% intervention rate
- Mid training: 10-20%
- Late training: 5-10%
- The offline buffer should maintain enough samples for meaningful learning

### Q: Can I pause training?
**A:** Yes, press Ctrl+C to safely stop. The policy is saved to `output-dir/final_policy`.

### Q: How do I resume training?
**A:** Currently manual - load the saved policy and restart training with lower initial intervention.

## References

- [HIL-SERL Paper](https://arxiv.org/abs/2312.08960)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [Piper SDK](https://github.com/agilexrobotics/piper_sdk)
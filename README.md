# roboplan-lerobot

[RoboPlan](https://github.com/open-planning/roboplan) inside a [LeRobot](https://github.com/huggingface/lerobot) imitation learning workflow, end to end in MuJoCo.

RoboPlan's **OInK** solver (Optimal Inverse Kinematics) sits between the operator or the learned policy and the simulated SO-101 arm:

- **Teleop**: keyboard end-effector commands become `FrameTask` targets; OInK produces the joint motion.
- **Datasets in either space**: joint positions (the real-SO-101 schema) or gripper-frame poses.
- **Policy execution**: a diffusion (or ACT) policy's actions are tracked through the matching loadout, `ConfigurationTask` for joint actions, `FrameTask` for Cartesian ones.
- **Barriers everywhere**: joint limits, a self-collision barrier and a table barrier apply to teleop and policy alike.

The sim is plugged in through LeRobot's own `Robot` and `Teleoperator` interfaces, so the scripts look like the ones you would use with hardware.

## Install

```bash
# install pixi first: https://pixi.sh
git clone https://github.com/sea-bass/roboplan-lerobot.git
cd roboplan-lerobot
pixi install           # roboplan from conda-forge, lerobot from PyPI; nothing to compile
pixi run so101-check   # RoboPlan FK vs. MuJoCo, both OInK loadouts, the table barrier
```

Linux and an NVIDIA GPU (8 GB is enough) are needed for training.
`pixi task list` shows every command.

## Guide

[Imitation learning with OInK in the loop](docs/imitation_learning.md): teleoperate, record, train, roll out.
The code lives in [`so101_oink_demo/`](so101_oink_demo).

# roboplan-lerobot

This package wires [RoboPlan](https://github.com/sea-bass/roboplan) into [LeRobot](https://github.com/huggingface/lerobot) to give you an end-to-end imitation learning workflow in simulation. You teleoperate a Kinova Gen3 arm in MuJoCo, the demonstrations get saved in LeRobot's dataset format, and you train an ACT policy on them. At inference time the policy outputs Cartesian targets which roboplan's OInK solver turns back into joint commands — so the learned policy never has to think about IK. Everything runs inside Docker, no ROS required.

## Quickstart

```bash
make build   # ~15-20 min the first time
make run     # open a shell in the container
make test    # verify all dependencies
```

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

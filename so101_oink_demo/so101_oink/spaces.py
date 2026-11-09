"""The two action spaces a dataset can be recorded in.

- ``joint``: absolute joint targets in degrees (LeRobot's real-SO-101 convention).
- ``cartesian``: absolute pose of the gripper frame in the robot base frame, as
  xyz in meters plus the first two columns of the rotation matrix (a continuous
  "6D" rotation), and the gripper joint in degrees.

The observation always carries the joint positions and the gripper-frame pose.
"""

from typing import Literal

import numpy as np

from so101_oink.env import JOINT_NAMES

ActionSpace = Literal["joint", "cartesian"]

JOINT_KEYS = [f"{joint}.pos" for joint in JOINT_NAMES]
POSE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.r1x", "ee.r1y", "ee.r1z", "ee.r2x", "ee.r2y", "ee.r2z"]
GRIPPER_KEY = "gripper.pos"
ACTION_KEYS = {"joint": JOINT_KEYS, "cartesian": POSE_KEYS + [GRIPPER_KEY]}


def pose_to_dict(tform: np.ndarray) -> dict[str, float]:
    vec = np.concatenate([tform[:3, 3], tform[:3, 0], tform[:3, 1]])
    return {key: float(v) for key, v in zip(POSE_KEYS, vec)}


def dict_to_pose(action: dict) -> np.ndarray:
    """Rebuilds a valid rotation (Gram-Schmidt): a policy never outputs an exact one."""
    vec = np.array([action[key] for key in POSE_KEYS])
    b1 = vec[3:6] / np.linalg.norm(vec[3:6])
    b2 = vec[6:9] - np.dot(b1, vec[6:9]) * b1
    b2 /= np.linalg.norm(b2)
    tform = np.eye(4)
    tform[:3, :3] = np.column_stack([b1, b2, np.cross(b1, b2)])
    tform[:3, 3] = vec[:3]
    return tform


def joints_to_dict(q_rad: np.ndarray) -> dict[str, float]:
    return {key: float(np.degrees(v)) for key, v in zip(JOINT_KEYS, q_rad)}


def dict_to_joints(action: dict) -> np.ndarray:
    return np.radians([action[key] for key in JOINT_KEYS])

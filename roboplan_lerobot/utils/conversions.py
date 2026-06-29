"""MuJoCo / Pinocchio coordinate conversions for the Kinova Gen3."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

# Arm joints that are continuous (revolves past 2*pi) in the URDF.
# Pinocchio represents them as [cos, sin] instead of a raw angle.
CONTINUOUS: set[int] = {0, 2, 4, 6}


def mj_q_to_pin(arm_q: np.ndarray) -> np.ndarray:
    """Convert 7 MuJoCo arm joint angles to 11-D Pinocchio config.

    Continuous joints expand from one angle to [cos, sin].
    Revolute joints pass through unchanged.
    """
    out: list[float] = []
    for i, angle in enumerate(arm_q):
        if i in CONTINUOUS:
            out.extend([np.cos(angle), np.sin(angle)])
        else:
            out.append(float(angle))
    return np.array(out)


def pin_q_to_mj(q_full: np.ndarray) -> np.ndarray:
    """Convert the leading arm-joint block of a Pinocchio config to 7 MuJoCo angles.

    Reads from q_full as returned by scene.integrate().
    Continuous joints are recovered via arctan2.
    """
    out: list[float] = []
    idx: int = 0
    for i in range(7):
        if i in CONTINUOUS:
            out.append(float(np.arctan2(q_full[idx + 1], q_full[idx])))
            idx += 2
        else:
            out.append(float(q_full[idx]))
            idx += 1
    return np.array(out)


def pose_to_transform(action: np.ndarray) -> np.ndarray:
    """Convert [pos(3), quat_xyzw(4)] to a 4x4 homogeneous transform."""
    x, y, z, w = float(action[3]), float(action[4]), float(action[5]), float(action[6])
    T: np.ndarray = np.eye(4)
    T[:3, :3] = pin.Quaternion(w, x, y, z).toRotationMatrix()
    T[:3, 3] = action[:3]
    return T

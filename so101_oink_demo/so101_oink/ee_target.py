"""Virtual end-effector target for the 5-DOF SO-101.

The arm has no wrist yaw, so the orientation target is kept consistent with
base pan: the home grasp orientation yawed toward the target position, then
pitched and rolled by the operator's offsets. Such targets are (nearly)
reachable, which keeps the FrameTask well-posed. Used by keyboard teleop and
by the scripted expert.
"""

import numpy as np
from scipy.spatial.transform import Rotation

WORKSPACE_LOW = np.array([0.12, -0.25, 0.0])
WORKSPACE_HIGH = np.array([0.38, 0.25, 0.35])
PITCH_RANGE = (-1.0, 1.0)  # rad, offset from the home grasp pitch
ROLL_RANGE = (-0.9, 0.9)  # rad, jaw-plane roll about the tool axis


class EETarget:
    def __init__(self):
        self.target_pos = np.zeros(3)
        self.pitch = 0.0
        self.roll = 0.0
        self._r_nominal = np.eye(3)

    def reset(self, pose: np.ndarray) -> None:
        """Re-anchor at an end-effector pose (typically FK of the current configuration)."""
        self.target_pos = pose[:3, 3].copy()
        yaw = np.arctan2(self.target_pos[1], self.target_pos[0])
        self._r_nominal = Rotation.from_euler("z", -yaw).as_matrix() @ pose[:3, :3]
        self.pitch = self.roll = 0.0

    def move(self, delta_pos: np.ndarray, delta_pitch: float = 0.0, delta_roll: float = 0.0) -> None:
        """Displace the target, clamped to the workspace box and the pitch/roll ranges."""
        self.target_pos = np.clip(self.target_pos + delta_pos, WORKSPACE_LOW, WORKSPACE_HIGH)
        self.pitch = float(np.clip(self.pitch + delta_pitch, *PITCH_RANGE))
        self.roll = float(np.clip(self.roll + delta_roll, *ROLL_RANGE))

    @property
    def target_rot(self) -> np.ndarray:
        yaw = np.arctan2(self.target_pos[1], self.target_pos[0])
        return (
            Rotation.from_euler("z", yaw).as_matrix()
            @ self._r_nominal
            @ Rotation.from_euler("YZ", [self.pitch, self.roll]).as_matrix()
        )

    def pose(self) -> np.ndarray:
        tform = np.eye(4)
        tform[:3, :3] = self.target_rot
        tform[:3, 3] = self.target_pos
        return tform

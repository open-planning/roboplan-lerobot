"""LeRobot processor steps that put OInK in the teleop pipeline.

    keyboard offsets --EETargetFromOffsets--> gripper-frame pose + gripper.pos
                     --OinkPoseToJoints-->    {joint}.pos            (joint datasets only)

For a Cartesian dataset the pipeline stops after the first step and the robot's
own FrameTask tracks the pose. For a joint dataset the second step converts the
pose to joint targets with a FrameTask of its own, so the recorded actions are
joint positions and the robot's ConfigurationTask tracks those.
"""

from dataclasses import dataclass, field

import numpy as np
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotActionProcessorStep,
    TransitionKey,
)
from so101_oink.ee_target import EETarget
from so101_oink.env import GRIPPER_OPEN, JOINT_NAMES
from so101_oink.oink_controller import OinkConfig, OinkController
from so101_oink.spaces import (
    ACTION_KEYS,
    GRIPPER_KEY,
    JOINT_KEYS,
    POSE_KEYS,
    dict_to_pose,
    pose_to_dict,
)

TELEOP_KEYS = ["enabled", "delta_x", "delta_y", "delta_z", "pitch", "roll", "gripper_vel"]
GRIPPER_RANGE = (-8.0, float(np.degrees(GRIPPER_OPEN)))  # degrees: firm close .. the home (open) position
GRIPPER_SPEED = 2.4  # degrees per tick at |gripper_vel| = 1


def _measured_joints(observation: dict) -> np.ndarray:
    return np.radians([observation[f"{joint}.pos"] for joint in JOINT_NAMES])


def _set_action_features(features, keys):
    """Replace the pipeline's action features by ``keys``, in order."""
    features[PipelineFeatureType.ACTION] = {key: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for key in keys}
    return features


@ProcessorStepRegistry.register("so101_oink_ee_target_from_offsets")
@dataclass
class EETargetFromOffsets(RobotActionProcessorStep):
    """Keyboard offsets -> absolute gripper-frame pose (yaw-coupled) + gripper position.

    The reference pose is latched from the measured joints (through RoboPlan's
    FK) when the teleop is first enabled after a reset; the gripper velocity is
    integrated on the measured gripper position.
    """

    _fk: OinkController = field(default_factory=lambda: OinkController("joint"), repr=False)
    _ee: EETarget = field(default_factory=EETarget, repr=False)
    _latched: bool = field(default=False, repr=False)
    _gripper: float | None = field(default=None, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition[TransitionKey.OBSERVATION]
        q = _measured_joints(observation)
        enabled = bool(action.pop("enabled"))
        delta = np.array([action.pop("delta_x"), action.pop("delta_y"), action.pop("delta_z")])
        pitch, roll, gripper_vel = action.pop("pitch"), action.pop("roll"), action.pop("gripper_vel")

        if not (self._latched and enabled):
            self._ee.reset(self._fk.fk(q))
            self._reference = self._ee.target_pos.copy()
            self._latched = enabled
        if enabled:  # EETarget.move clamps to the workspace box
            self._ee.move(self._reference + delta - self._ee.target_pos, pitch - self._ee.pitch, roll - self._ee.roll)
        if self._gripper is None:
            self._gripper = observation[GRIPPER_KEY]
        self._gripper = float(np.clip(self._gripper + GRIPPER_SPEED * gripper_vel, *GRIPPER_RANGE))

        return {**action, **pose_to_dict(self._ee.pose()), GRIPPER_KEY: self._gripper}

    def reset(self) -> None:
        self._latched = False
        self._gripper = None

    def transform_features(self, features):
        return _set_action_features(features, ACTION_KEYS["cartesian"])


@ProcessorStepRegistry.register("so101_oink_pose_to_joints")
@dataclass
class OinkPoseToJoints(RobotActionProcessorStep):
    """Gripper-frame pose -> joint targets in degrees, through an OInK FrameTask.

    Runs the solver for one policy tick toward the pose, as the robot would in
    the Cartesian loadout, and emits the commanded configuration. The solver is
    re-synchronized with the measured joints after every ``reset()``.
    """

    oink_config: OinkConfig = field(default_factory=OinkConfig)
    fps: int = 25
    _controller: OinkController | None = field(default=None, repr=False)
    _needs_sync: bool = field(default=True, repr=False)

    def __post_init__(self):
        self._controller = OinkController("cartesian", self.oink_config)
        self._steps = round(self.oink_config.control_freq / self.fps)

    def action(self, action: RobotAction) -> RobotAction:
        q = _measured_joints(self.transition[TransitionKey.OBSERVATION])
        if self._needs_sync:
            self._controller.reset(q)
            self._needs_sync = False
        self._controller.set_gripper(q[5])
        self._controller.set_pose_target(dict_to_pose(action))
        for _ in range(self._steps):
            q_arm = self._controller.step()
        joints = {key: float(v) for key, v in zip(JOINT_KEYS, np.degrees(q_arm))}
        return {**{k: v for k, v in action.items() if k not in POSE_KEYS}, **joints, GRIPPER_KEY: action[GRIPPER_KEY]}

    def reset(self) -> None:
        self._needs_sync = True

    def transform_features(self, features):
        return _set_action_features(features, ACTION_KEYS["joint"])

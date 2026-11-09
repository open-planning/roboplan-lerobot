"""LeRobot Robot plugin: the MuJoCo SO-101 driven by RoboPlan's OInK solver.

``send_action`` never writes the action to the motors. It becomes the target of
an OInK task, the solver runs 4 steps toward it under the joint limits and the
self-collision and table barriers, and the resulting joint commands drive
MuJoCo's position servos. Joint positions cross this boundary in degrees (the
LeRobot convention); MuJoCo and RoboPlan work in radians.
"""

from typing import Any

import numpy as np
from lerobot.robots import Robot
from so101_oink.env import CUBE_COLORS, JOINT_NAMES, TASKS, SO101PickCubeEnv
from so101_oink.oink_controller import OinkController
from so101_oink.spaces import (
    ACTION_KEYS,
    POSE_KEYS,
    dict_to_joints,
    dict_to_pose,
    pose_to_dict,
)

from .config_so101_oink import SO101OinkConfig


class SO101Oink(Robot):
    config_class = SO101OinkConfig
    name = "so101_oink"

    def __init__(self, config: SO101OinkConfig):
        super().__init__(config)
        self.config = config
        self.env: SO101PickCubeEnv | None = None
        self.controller: OinkController | None = None
        self.viewer = None
        self.cameras: dict = {}  # images come from the renderer, not physical cameras

    @property
    def observation_features(self) -> dict:
        h, w = self.config.image_height, self.config.image_width
        return {
            **{f"{joint}.pos": float for joint in JOINT_NAMES},
            **dict.fromkeys(POSE_KEYS, float),
            "wrist": (h, w, 3),
            "front": (h, w, 3),
        }

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(ACTION_KEYS[self.config.action_space], float)

    @property
    def is_connected(self) -> bool:
        return self.env is not None

    def connect(self, calibrate: bool = True) -> None:
        self.env = SO101PickCubeEnv(
            image_size=(self.config.image_height, self.config.image_width), target_color=self.config.target_color
        )
        self.env.reset(seed=self.config.seed)
        self.controller = OinkController(self.config.action_space, self.config.oink)
        self.controller.reset(self.env.joint_pos)
        self.solver_steps_per_tick = round(self.env.control_dt / self.controller.dt)
        if self.config.show_viewer:
            self.viewer = self._launch_viewer()

    def _launch_viewer(self):
        if self.config.viewer == "viser":
            from .viser_viewer import ViserViewer

            return ViserViewer(self.env.model, self.env.data, port=self.config.viser_port)
        import mujoco.viewer

        return mujoco.viewer.launch_passive(self.env.model, self.env.data, show_left_ui=False, show_right_ui=False)

    def disconnect(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.env is not None:
            self.env.close()
            self.env = None
        self.controller = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        q = self.env.joint_pos
        return {
            **{f"{joint}.pos": float(np.degrees(v)) for joint, v in zip(JOINT_NAMES, q)},
            **pose_to_dict(self.controller.fk(q)),
            "wrist": self.env.render_camera("wrist"),
            "front": self.env.render_camera("front"),
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Hand the action to OInK as a task target and track it for one policy tick."""
        if self.config.action_space == "joint":
            q_target = dict_to_joints(action)
            self.controller.set_joint_target(q_target[:5])
            gripper = q_target[5]
        else:
            self.controller.set_pose_target(dict_to_pose(action))
            gripper = np.radians(action["gripper.pos"])
        for _ in range(self.solver_steps_per_tick):
            self.controller.set_gripper(self.env.joint_pos[5])
            self.env.substep(np.append(self.controller.step(), gripper), self.controller.dt)
        if self.viewer is not None:
            self.viewer.sync()
        return action

    # ------------------------------------------------------------ sim utilities

    def reset_scene(self, seed: int | None = None, target_color: str | None = None) -> None:
        """Re-home the arm and respawn both cubes; ``target_color`` switches the task."""
        self.env.reset(seed=seed, options={"target_color": target_color} if target_color else None)
        self.controller.reset(self.env.joint_pos)
        if self.viewer is not None:
            self.viewer.sync()

    @property
    def ee_pose(self) -> np.ndarray:
        """4x4 world pose of the gripper frame (RoboPlan FK of the measured joints)."""
        return self.controller.fk(self.env.joint_pos)

    @property
    def target_color(self) -> str:
        return self.env.target_color

    @property
    def task(self) -> str:
        return TASKS[self.env.target_color]

    @property
    def is_success(self) -> bool:
        return self.env._is_success()

    @property
    def wrong_cube_lifted(self) -> bool:
        return any(self.env.is_lifted(c) for c in CUBE_COLORS if c != self.env.target_color)

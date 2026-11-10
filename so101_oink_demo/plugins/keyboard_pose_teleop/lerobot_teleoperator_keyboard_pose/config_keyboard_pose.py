from dataclasses import dataclass
from typing import Literal

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard_pose")
@dataclass
class KeyboardPoseConfig(TeleoperatorConfig):
    """Keyboard end-effector teleoperator.

    Emits offsets from a reference pose that the OInK processor step latches when
    the teleop is enabled: delta_x/y/z (m), pitch and roll (rad), and a gripper
    velocity. Keys come from the terminal, or with ``key_source=browser`` from the
    viser viewer's tab.
    """

    move_speed: float = 0.15  # m/s
    pitch_speed: float = 1.2  # rad/s
    gripper_vel: float = 1.0  # magnitude handed to the gripper integrator
    smoothing: float = 0.1  # s, time constant of the rate filter (0 = none)
    key_source: Literal["terminal", "browser"] = "terminal"
    browser_port: int = 8081

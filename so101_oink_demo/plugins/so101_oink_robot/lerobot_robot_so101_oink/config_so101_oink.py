from dataclasses import dataclass, field
from typing import Literal

from lerobot.robots import RobotConfig
from so101_oink.oink_controller import OinkConfig

VIEWERS = ("native", "viser")


@RobotConfig.register_subclass("so101_oink")
@dataclass
class SO101OinkConfig(RobotConfig):
    """SO-101 in MuJoCo, commanded through RoboPlan's OInK solver.

    ``action_space`` picks the OInK loadout and the action schema (see
    ``so101_oink.spaces``); ``oink`` holds the solver settings, e.g.
    ``--robot.oink.table_z=0.02``. The observation is the joint positions in
    degrees, the gripper-frame pose, and two RGB cameras (``wrist``, ``front``).
    ``viewer=viser`` serves the scene to a browser instead of a MuJoCo window.
    """

    action_space: Literal["joint", "cartesian"] = "joint"
    oink: OinkConfig = field(default_factory=OinkConfig)
    image_height: int = 240
    image_width: int = 320
    show_viewer: bool = True
    viewer: Literal["native", "viser"] = "native"
    viser_port: int = 8080
    seed: int | None = None
    target_color: str = "red"

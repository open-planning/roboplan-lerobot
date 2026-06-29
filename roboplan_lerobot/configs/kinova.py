from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from roboplan.example_models import get_package_share_dir

_KINOVA_MODELS = (
    Path(get_package_share_dir()) / "roboplan_example_models" / "models" / "kinova_robot_model"
)
_ASSETS = Path(__file__).parent.parent / "assets"


@dataclass
class KinovaEnvConfig:
    """Config for KinovaReach-v0. Set scene_mjcf_path=None to load the bare robot."""

    mjcf_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("MUJOCO_MENAGERIE_PATH", "")
        ) / "kinova_gen3" / "gen3.xml"
    )
    # Worldbody children from this file are merged into mjcf_path at load time.
    scene_mjcf_path: Path | None = field(
        default_factory=lambda: _ASSETS / "scene_objects.xml"
    )
    urdf_path: Path = field(default_factory=lambda: _KINOVA_MODELS / "kinova_robotiq.urdf")
    srdf_path: Path = field(default_factory=lambda: _KINOVA_MODELS / "kinova_robotiq.srdf")
    yaml_config_path: Path = field(
        default_factory=lambda: _KINOVA_MODELS / "kinova_robotiq_config.yaml"
    )
    control_freq: float = 50.0
    viser_host: str = "0.0.0.0"
    viser_port: int = 8080

    def __post_init__(self) -> None:
        if not self.mjcf_path.exists():
            env_val = os.environ.get("MUJOCO_MENAGERIE_PATH")
            if env_val is None:
                raise EnvironmentError(
                    "MUJOCO_MENAGERIE_PATH is not set.\n"
                    "Clone https://github.com/google-deepmind/mujoco_menagerie "
                    "and set:\n"
                    "  export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie"
                )
            raise FileNotFoundError(
                f"MuJoCo model not found at {self.mjcf_path}.\n"
                f"Check that MUJOCO_MENAGERIE_PATH={env_val!r} contains "
                "kinova_gen3/gen3.xml."
            )

"""Gymnasium environment: SO-101 arm picking one of two cubes in MuJoCo.

Each episode places the red and green cube in random slots; the episode's
target color decides which one must be lifted. Actions are absolute joint
targets in radians (5 arm joints + gripper) and all inverse kinematics, limits
and barriers live in the OInK controller that produces them.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_XML = Path(__file__).parent / "model" / "scene.xml"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_JOINTS = JOINT_NAMES[:5]
EE_SITE = "gripperframe"  # coincides with the URDF's gripper_frame_link

# the "home" keyframe of the MJCF scene (radians)
HOME_Q = np.array([0.0, -0.5, 0.8, 0.6, 0.0, 0.5])

GRIPPER_OPEN = 1.3
GRIPPER_CLOSED = -0.1

CUBE_COLORS = ("red", "green")
CUBE_HALF = 0.0125

# language instruction recorded with each episode (the target color decides it)
TASKS = {color: f"Pick up the {color} cube and lift it." for color in CUBE_COLORS}


class SO101PickCubeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        image_size: tuple[int, int] = (240, 320),
        control_dt: float = 0.04,
        # Spawn slots (base frame: +x forward, +y left). Both cubes share the x
        # range; the y ranges are separated so the open jaws of the gripper
        # never reach the other cube while grasping.
        cube_spawn_x: tuple[float, float] = (0.20, 0.28),
        cube_spawn_y_right: tuple[float, float] = (-0.09, -0.05),
        cube_spawn_y_left: tuple[float, float] = (0.0, 0.04),
        target_color: str = "red",
        lift_height: float = 0.08,
    ):
        super().__init__()
        self.image_size = image_size
        self.lift_height = lift_height
        self.cube_spawn_x = cube_spawn_x
        self.slot_spawn_y = {"right": cube_spawn_y_right, "left": cube_spawn_y_left}
        self.target_color = target_color

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = round(control_dt / self.model.opt.timestep)
        self.control_dt = self.n_substeps * self.model.opt.timestep

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.qpos_ids = [self.model.jnt_qposadr[j] for j in self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self._cube_qpos_adr = {
            color: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{color}_free")
            ]
            for color in CUBE_COLORS
        }
        self.cube_sides: dict[str, str] = {}
        self.home_keyframe = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._renderer = None

    # ------------------------------------------------------------------ helpers

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            h, w = self.image_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderer

    def render_camera(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    @property
    def joint_pos(self) -> np.ndarray:
        """(6,) measured joint positions in radians."""
        return self.data.qpos[self.qpos_ids].astype(np.float32)

    @property
    def home_joint_pos(self) -> np.ndarray:
        """(6,) joint positions of the ``home`` keyframe in radians."""
        return self.model.key_qpos[self.home_keyframe][self.qpos_ids].astype(np.float64)

    @property
    def target_color(self) -> str:
        return self._target_color

    @target_color.setter
    def target_color(self, color: str) -> None:
        if color not in CUBE_COLORS:
            raise ValueError(f"target_color must be one of {CUBE_COLORS}, got {color!r}")
        self._target_color = color

    @property
    def cube_qpos_adr(self) -> int:
        """qpos address of the target cube's free joint."""
        return self._cube_qpos_adr[self.target_color]

    def cube_pos_of(self, color: str) -> np.ndarray:
        adr = self._cube_qpos_adr[color]
        return self.data.qpos[adr : adr + 3].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        """Position of the target cube."""
        return self.cube_pos_of(self.target_color)

    def nudge_cube(self, delta_xy: np.ndarray) -> None:
        """Slide the target cube along the table (perturbation for recovery demonstrations)."""
        adr = self.cube_qpos_adr
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{self.target_color}_free")
        dof = self.model.jnt_dofadr[joint]
        self.data.qpos[adr : adr + 2] += delta_xy
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    @property
    def ee_pose(self) -> np.ndarray:
        """4x4 world pose of the gripper frame site, from the MuJoCo state."""
        tform = np.eye(4)
        tform[:3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        tform[:3, 3] = self.data.site_xpos[self.ee_site_id]
        return tform

    def is_lifted(self, color: str) -> bool:
        cube = self.cube_pos_of(color)
        lifted = cube[2] > self.lift_height
        held = np.linalg.norm(cube - self.ee_pos) < 0.08
        return bool(lifted and held)

    def _is_success(self) -> bool:
        return self.is_lifted(self.target_color)

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        """Re-home the arm and respawn both cubes.

        ``options={"target_color": "green"}`` switches the target for this and
        the following episodes.
        """
        super().reset(seed=seed)
        if options and "target_color" in options:
            self.target_color = options["target_color"]
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_keyframe)
        # random side assignment, then a random pose within each slot
        sides = ["left", "right"]
        self.np_random.shuffle(sides)
        self.cube_sides = dict(zip(CUBE_COLORS, sides))
        for color, side in self.cube_sides.items():
            x = self.np_random.uniform(*self.cube_spawn_x)
            y = self.np_random.uniform(*self.slot_spawn_y[side])
            yaw = self.np_random.uniform(-np.pi / 4, np.pi / 4)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            adr = self._cube_qpos_adr[color]
            self.data.qpos[adr : adr + 3] = [x, y, CUBE_HALF]
            self.data.qpos[adr + 3 : adr + 7] = quat
        mujoco.mj_forward(self.model, self.data)
        return self.joint_pos, self._info()

    def _info(self) -> dict:
        return {
            "target_color": self.target_color,
            "cube_pos": self.cube_pos,
            "cube_sides": dict(self.cube_sides),
            "ee_pos": self.ee_pos,
        }

    def substep(self, ctrl: np.ndarray, duration: float) -> None:
        """Hold ``ctrl`` (6 joint targets, rad) and advance the physics by ``duration``.

        The OInK controller runs faster than the policy (e.g. 100 Hz vs 25 Hz);
        each solver step calls this with its freshly integrated joint command.
        """
        self.data.ctrl[:] = np.clip(ctrl, self.action_space.low, self.action_space.high)
        for _ in range(max(1, round(duration / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        """One policy tick (``control_dt``) with a constant joint target."""
        self.substep(action, self.control_dt)
        success = self._is_success()
        reward = 1.0 if success else 0.0
        info = {"is_success": success, **self._info()}
        return self.joint_pos, reward, success, False, info

    def render(self):
        return self.render_camera("front")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

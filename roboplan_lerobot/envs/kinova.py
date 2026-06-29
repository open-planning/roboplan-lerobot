from __future__ import annotations

from typing import Literal

import gymnasium as gym
import mujoco
import numpy as np
import pinocchio as pin
import xacro
from gymnasium import spaces
from gymnasium.utils import seeding
from pinocchio.visualize import ViserVisualizer
from roboplan.core import CartesianConfiguration, Scene
from roboplan.example_models import get_package_share_dir
from roboplan.optimal_ik import (
    FrameTask,
    FrameTaskOptions,
    Oink,
    PositionLimit,
    VelocityLimit,
)

from roboplan_lerobot.configs.kinova import KinovaEnvConfig
from roboplan_lerobot.utils import build_scene_xml, cylinder_mesh, mj_q_to_pin, pin_q_to_mj, pose_to_transform

JOINT_GROUP: str = "manipulator"
BASE_LINK: str = "base_link"
EE_FRAME: str = "robotiq_85_base_link"

# Home joint angles in MuJoCo convention (radians).
HOME_QPOS: np.ndarray = np.array([0.0, 0.2618, np.pi, -2.2689, 0.0, 0.96, np.pi / 2])

RenderMode = Literal["viser", "rgb_array", None]


class KinovaEnv(gym.Env):
    """Kinova Gen3 environment backed by MuJoCo and roboplan OInK.

    Actions are 7-D: target EE position (3) + quaternion xyzw (4) in world frame.
    OInK solves the IK and updates joint positions each step.

    Observations are 15-D: EE position (3), EE quaternion xyzw (4),
    arm joint angles (7), gripper opening (1, always 0 for now).

    render_mode="viser" starts a 3D viewer at http://localhost:8080.
    render_mode="rgb_array" renders offscreen via env.render().
    render_mode=None disables rendering.
    """

    metadata: dict = {"render_modes": ["viser", "rgb_array"], "render_fps": 50}

    def __init__(self, config: KinovaEnvConfig, render_mode: RenderMode = None) -> None:
        self.config: KinovaEnvConfig = config
        self.render_mode: RenderMode = render_mode
        self.dt: float = 1.0 / self.config.control_freq

        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_string(
            build_scene_xml(self.config.mjcf_path, self.config.scene_mjcf_path)
        )
        self.mj_data: mujoco.MjData = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.dt
        self._renderer: mujoco.Renderer | None = None

        package_paths: list[str] = [get_package_share_dir()]
        urdf_xml: str = xacro.process_file(self.config.urdf_path).toxml()
        srdf_xml: str = xacro.process_file(self.config.srdf_path).toxml()
        self.scene: Scene = Scene(
            "kinova_env",
            urdf=urdf_xml,
            srdf=srdf_xml,
            package_paths=package_paths,
            yaml_config_path=self.config.yaml_config_path,
        )
        # nv includes mimic gripper joints; needed for scene.integrate().
        self._nv: int = pin.buildModelFromXML(urdf_xml, mimic=True).nv

        self.oink: Oink = Oink(self.scene, JOINT_GROUP)
        joint_names: list[str] = self.scene.getJointGroupInfo(JOINT_GROUP).joint_names
        v_max: np.ndarray = np.hstack([
            self.scene.getJointInfo(n).limits.max_velocity for n in joint_names
        ])
        self._constraints: list = [
            PositionLimit(self.oink, gain=1.0),
            VelocityLimit(self.oink, self.dt, v_max),
        ]

        goal: CartesianConfiguration = CartesianConfiguration()
        goal.base_frame = BASE_LINK
        goal.tip_frame = EE_FRAME
        self._frame_task: FrameTask = FrameTask(
            self.oink,
            self.scene,
            goal,
            FrameTaskOptions(
                position_cost=1.0,
                orientation_cost=0.1,
                task_gain=1.0,
                lm_damping=0.01,
            ),
        )

        self._viz: ViserVisualizer | None = None
        self._viz_q: np.ndarray | None = None
        self._viser_obj_handles: dict = {}
        if render_mode == "viser":
            model_pin = pin.buildModelFromXML(urdf_xml, mimic=True)
            collision_model = pin.buildGeomFromUrdfString(
                model_pin, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
            )
            visual_model = pin.buildGeomFromUrdfString(
                model_pin, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
            )
            self._viz = ViserVisualizer(model_pin, collision_model, visual_model)
            self._viz.initViewer(
                open=False,
                loadModel=True,
                host=self.config.viser_host,
                port=self.config.viser_port,
            )
            self._setup_viser_scene()
            print(f"Viser viewer: http://localhost:{self.config.viser_port}")

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:7] = HOME_QPOS
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._sync_to_scene()
        self._viz_update()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        q_current: np.ndarray = self.scene.getCurrentJointPositions()

        base_T_world: np.ndarray = np.linalg.inv(
            self.scene.forwardKinematics(q_current, BASE_LINK)
        )
        self._frame_task.setTargetFrameTransform(
            base_T_world @ pose_to_transform(action)
        )

        delta_q: np.ndarray = np.zeros(len(self.oink.v_indices))
        ik_failed: bool = False
        try:
            self.oink.solveIk(
                self.scene, [self._frame_task], self._constraints, [], delta_q, 1e-6
            )
        except RuntimeError:
            ik_failed = True

        delta_q_full: np.ndarray = np.zeros(self._nv)
        delta_q_full[self.oink.v_indices] = delta_q
        q_new: np.ndarray = self.scene.integrate(q_current, delta_q_full)
        self.scene.setJointPositions(q_new)
        self.scene.forwardKinematics(q_new, EE_FRAME)

        self.mj_data.qpos[:7] = pin_q_to_mj(q_new)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._viz_q = q_new
        self._viz_update()

        return self._get_obs(), 0.0, False, False, {"ik_failed": ik_failed}

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.mj_model)
            self._renderer.update_scene(self.mj_data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viz is not None:
            self._viz.viewer.stop()
            self._viz = None

    def _get_obs(self) -> np.ndarray:
        q: np.ndarray = self.scene.getCurrentJointPositions()
        ee_T: np.ndarray = self.scene.forwardKinematics(q, EE_FRAME)
        ee_pos: np.ndarray = ee_T[:3, 3]
        ee_quat_xyzw: np.ndarray = pin.Quaternion(ee_T[:3, :3]).coeffs()
        joint_angles: np.ndarray = self.mj_data.qpos[:7].copy()
        gripper: np.ndarray = np.array([0.0])  # TODO: implement gripper actuation
        return np.concatenate([ee_pos, ee_quat_xyzw, joint_angles, gripper]).astype(np.float32)

    def _sync_to_scene(self) -> None:
        """Copy MuJoCo arm joint state into the roboplan scene and run FK."""
        q_current: np.ndarray = self.scene.getCurrentJointPositions()
        q_current[self.oink.q_indices] = mj_q_to_pin(self.mj_data.qpos[:7])
        self.scene.setJointPositions(q_current)
        self.scene.forwardKinematics(q_current, EE_FRAME)
        self._viz_q = q_current

    def _setup_viser_scene(self) -> None:
        """Add the table and pick objects to viser.

        Table geometry is static. Pick objects store handles so _viz_update
        can track their positions from mj_data each step.
        """
        server = self._viz.viewer

        # viser 1.x add_box takes full dimensions, not half extents
        _WOOD = (209, 181, 141)
        _LEG  = (153, 128, 102)
        server.scene.add_box(
            "/scene/table/top",
            dimensions=(1.0, 1.0, 0.04),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.25, 0.0, -0.02),
            color=_WOOD,
        )
        for tag, pos in (
            ("fl", (-0.20, -0.45, -0.39)),
            ("fr", ( 0.70, -0.45, -0.39)),
            ("bl", (-0.20,  0.45, -0.39)),
            ("br", ( 0.70,  0.45, -0.39)),
        ):
            server.scene.add_box(
                f"/scene/table/leg_{tag}",
                dimensions=(0.06, 0.06, 0.70),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=pos,
                color=_LEG,
            )

        cyl_v, cyl_f = cylinder_mesh(0.025, 0.08)
        cap_v, cap_f = cylinder_mesh(0.02,  0.10)

        _objects = (
            ("red_cube",       "box",  {"dimensions": (0.05, 0.05, 0.05)},  (0.40,  0.15, 0.025), (217,  51,  51)),
            ("blue_mug",       "mesh", {"vertices": cyl_v, "faces": cyl_f}, (0.50, -0.12, 0.040), ( 51,  89, 217)),
            ("yellow_capsule", "mesh", {"vertices": cap_v, "faces": cap_f}, (0.35, -0.08, 0.050), (230, 204,  26)),
        )
        for body_name, kind, kwargs, pos, color in _objects:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            common = dict(wxyz=(1.0, 0.0, 0.0, 0.0), position=pos, color=color)
            if kind == "box":
                handle = server.scene.add_box(f"/scene/objects/{body_name}", **kwargs, **common)
            else:
                handle = server.scene.add_mesh_simple(f"/scene/objects/{body_name}", **kwargs, **common)
            self._viser_obj_handles[body_name] = (handle, body_id)

    def _viz_update(self) -> None:
        if self._viz is not None and self._viz_q is not None:
            self._viz.display(self._viz_q)
            for _name, (handle, body_id) in self._viser_obj_handles.items():
                handle.position = tuple(self.mj_data.xpos[body_id])
                handle.wxyz = tuple(self.mj_data.xquat[body_id])

"""The OInK loadouts that turn joint-space or Cartesian-space actions into joint commands.

OInK solves one small QP per control step for a joint displacement that pursues
tasks under constraints (joint position and velocity limits) and barriers (keep
out of self-collision, keep the gripper above the table). The action space picks
the primary task:

- ``joint``:     ConfigurationTask tracks an absolute joint target.
- ``cartesian``: FrameTask tracks an absolute pose of the gripper frame, with a
                 low-weight ConfigurationTask toward the home posture in its nullspace.

The solver integrates its own commanded configuration (a virtual robot that the
simulated servos follow) and is re-synchronized to the measured state on reset.
Only the 5 arm joints are solved for; the gripper is commanded directly but kept
in the solver's configuration so the moving jaw takes part in collision checking.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from roboplan.core import (
    CartesianConfiguration,
    Scene,
    loadJointLimitsConfig,
    loadTextFile,
    loadUrdfSceneDescriptionFromXml,
)
from roboplan.example_models import get_package_models_dir, get_package_share_dir
from roboplan.optimal_ik import (
    ConfigurationTask,
    ConfigurationTaskOptions,
    ConstraintAxisSelection,
    FrameTask,
    FrameTaskOptions,
    Oink,
    PositionBarrier,
    PositionLimit,
    SelfCollisionBarrier,
    SelfCollisionBarrierOptions,
    VelocityLimit,
)

from so101_oink.collision_model import box_collision_urdf
from so101_oink.env import ARM_JOINTS, HOME_Q, JOINT_NAMES

ActionSpace = Literal["joint", "cartesian"]

MODEL_DIR = get_package_models_dir() / "so101_robot_model"
ARM_GROUP = "arm"  # SRDF chain base_link -> gripper_link (5 joints)
EE_FRAME = "gripper_frame_link"  # same placement as the MJCF ``gripperframe`` site
TABLE_FRAMES = (EE_FRAME, "moving_jaw_so101_v1_link")  # kept above the table


@dataclass
class OinkConfig:
    # Solver rate: 4 steps per 25 Hz policy tick.
    control_freq: float = 100.0
    # Joint velocity cap (rad/s).
    max_joint_velocity: float = 3.0
    # Fraction of the remaining task error closed per solver step. 1.0 is dead-beat:
    # a new 25 Hz target is reached in the first 10 ms step and the arm then holds,
    # which turns policy actions into a visible staircase; 0.25 spreads it over the tick.
    task_gain: float = 0.25
    # FrameTask weights. Position dominates: a 5-DOF arm cannot reach arbitrary
    # orientations, so orientation is tracked as well as the kinematics allow.
    position_cost: float = 1.0
    orientation_cost: float = 0.1
    # Self-collision barrier: keep the closest link pairs this far apart (m).
    self_collision: bool = True
    self_collision_distance: float = 0.01
    # Table barrier: the gripper frame and moving jaw stay above this world z (m).
    table_barrier: bool = True
    table_z: float = 0.0


def build_scene() -> Scene:
    """RoboPlan scene of the SO-101 from the URDF, SRDF and limits shipped with RoboPlan."""
    share_dir = get_package_share_dir()
    urdf = box_collision_urdf(MODEL_DIR / "so101.urdf", share_dir)
    scene = Scene("so101", loadUrdfSceneDescriptionFromXml(urdf, [share_dir]))
    scene.importSrdf(loadTextFile(MODEL_DIR / "so101.srdf"))
    scene.importJointLimitsFromConfig(loadJointLimitsConfig(MODEL_DIR / "so101_config.yaml"))
    # The moving jaw hinges millimeters from the wrist housing; the SRDF only disables
    # adjacent pairs, so this pair would keep the self-collision barrier permanently active.
    scene.setCollisions("wrist_link", "moving_jaw_so101_v1_link", False)
    return scene


class OinkController:
    def __init__(self, action_space: ActionSpace, config: OinkConfig | None = None):
        self.action_space = action_space
        self.config = cfg = config or OinkConfig()
        self.dt = 1.0 / cfg.control_freq

        self.scene = build_scene()
        self.q_idx = np.asarray(self.scene.getJointPositionIndices(JOINT_NAMES))
        self.arm_q_idx = np.asarray(self.scene.getJointPositionIndices(ARM_JOINTS))
        self.gripper_q_idx = int(self.scene.getJointPositionIndices(["gripper"])[0])
        self.q_full = self.scene.getCurrentJointPositions()
        self.q_full[self.q_idx] = HOME_Q
        self.home_q = self.q_full.copy()

        self.oink = Oink(self.scene, ARM_GROUP)
        nv = self.oink.num_variables
        self.delta_q = np.zeros(nv)
        self.delta_q_full = np.zeros(len(self.scene.getJointVelocityIndices(self.scene.getJointNames())))

        v_max = np.minimum(
            np.hstack([self.scene.getJointInfo(j).limits.max_velocity for j in self.oink.joint_names]),
            cfg.max_joint_velocity,
        )
        self.constraints = [PositionLimit(self.oink), VelocityLimit(self.oink, self.dt, v_max)]

        # Barriers are control barrier functions: inequality constraints that limit how fast
        # the arm may approach a boundary as a function of its distance to it. RoboPlan's
        # default safe_displacement_gain (1.0) adds a damping term far larger than the task
        # weights for this small arm, so it is zeroed to keep the barriers pure constraints.
        self.barriers = []
        self.self_collision_barrier = None
        self.table_barriers = []
        if cfg.self_collision:
            self.self_collision_barrier = SelfCollisionBarrier(
                self.oink,
                self.scene,
                self.dt,
                SelfCollisionBarrierOptions(
                    n_collision_pairs=3,
                    gain=10.0,
                    safe_displacement_gain=0.0,
                    d_min=cfg.self_collision_distance,
                    d_max=0.1,
                ),
            )
            self.barriers.append(self.self_collision_barrier)
        if cfg.table_barrier:
            for frame in TABLE_FRAMES:
                barrier = PositionBarrier(
                    self.oink, self.scene, frame,
                    p_min=np.array([-10.0, -10.0, cfg.table_z]), p_max=np.array([10.0, 10.0, 10.0]), dt=self.dt,
                    axis_selection=ConstraintAxisSelection(x=False, y=False, z=True),
                    gain=5.0, safe_displacement_gain=0.0,
                )  # fmt: skip
                self.table_barriers.append(barrier)
                self.barriers.append(barrier)

        q_group = self.home_q[self.oink.q_indices]
        if action_space == "joint":
            self.config_task = ConfigurationTask(
                self.oink, q_group, np.ones(nv), ConfigurationTaskOptions(task_gain=cfg.task_gain, lm_damping=0.01)
            )
            self.frame_task = None
            self.tasks = [self.config_task]
        else:
            self.frame_task = FrameTask(
                self.oink, self.scene, CartesianConfiguration("base_link", EE_FRAME, np.eye(4)),
                FrameTaskOptions(
                    position_cost=cfg.position_cost, orientation_cost=cfg.orientation_cost, task_gain=cfg.task_gain,
                    lm_damping=1.0, position_saturation=0.05, orientation_saturation=0.5,
                ),
            )  # fmt: skip
            self.config_task = ConfigurationTask(
                self.oink, q_group, np.full(nv, 1e-2), ConfigurationTaskOptions(task_gain=0.1, priority=2)
            )
            self.tasks = [self.frame_task, self.config_task]
        self.solver_failures = 0

    # ------------------------------------------------------------------------- state

    def fk(self, q: np.ndarray, frame: str = EE_FRAME) -> np.ndarray:
        """4x4 world pose of ``frame`` for (6,) joint positions in radians."""
        q_full = self.q_full.copy()
        q_full[self.q_idx] = q
        return self.scene.forwardKinematics(q_full, frame)

    @property
    def q_arm(self) -> np.ndarray:
        """(5,) commanded arm joint positions, radians."""
        return self.q_full[self.arm_q_idx].copy()

    def reset(self, q_measured: np.ndarray) -> None:
        """Re-synchronize the virtual robot with the measured (6,) joint state."""
        self.q_full[self.q_idx] = q_measured
        self.scene.setJointPositions(self.q_full)
        self.delta_q[:] = 0.0
        if self.frame_task is not None:
            self.set_pose_target(self.fk(q_measured))
        else:
            self.set_joint_target(q_measured[:5])

    def set_gripper(self, gripper_rad: float) -> None:
        self.q_full[self.gripper_q_idx] = gripper_rad

    # ----------------------------------------------------------------------- control

    def set_joint_target(self, q_arm: np.ndarray) -> None:
        q_target = self.q_full.copy()
        q_target[self.arm_q_idx] = q_arm
        self.config_task.setTargetConfiguration(q_target[self.oink.q_indices])

    def set_pose_target(self, world_T_target: np.ndarray) -> None:
        self.target_pose = np.asarray(world_T_target, dtype=np.float64)
        base_T_world = np.linalg.inv(self.scene.forwardKinematics(self.q_full, "base_link"))
        self.frame_task.setTargetFrameTransform(base_T_world @ self.target_pose)

    def step(self) -> np.ndarray:
        """One solver step toward the current target; returns the new commanded arm q (5,)."""
        try:
            self.oink.solveIk(self.q_full, self.tasks, self.constraints, self.barriers, self.delta_q, 1e-6)
        except RuntimeError:
            self.solver_failures += 1
            self.delta_q[:] = 0.0
        self.delta_q_full[self.oink.v_indices] = self.delta_q
        # The barriers are linearized; drop the step if forward kinematics says it still violates one.
        self.oink.enforceBarriers(self.q_full, self.barriers, self.delta_q_full)
        self.q_full = self.scene.integrate(self.q_full, self.delta_q_full)
        self.scene.setJointPositions(self.q_full)
        return self.q_arm

    def barrier_report(self) -> dict[str, float]:
        """Current barrier values in meters (>= 0 is safe), for logging."""
        report = {}
        if self.self_collision_barrier is not None:
            values = np.asarray(self.self_collision_barrier.barrier_values)
            report["link_clearance"] = float(values.min()) if values.size else np.inf
        for frame, barrier in zip(TABLE_FRAMES, self.table_barriers):
            report[f"{frame}_above_table"] = float(self.fk(self.q_full[self.q_idx], frame)[2, 3] - self.config.table_z)
        return report

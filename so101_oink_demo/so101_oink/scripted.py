"""Scripted expert for the pick-cube task.

``scripted_pick`` yields one command per policy tick: the target pose of the
gripper frame (4x4, world) and the gripper joint target (radians). The caller
runs it through OInK and records whichever action space its dataset uses.

Grasp strategy (found empirically):
    1. pitch the gripper steep (near-vertical beak) and hover above the cube,
       offset slightly *behind* it so the fixed jaw clears the cube on descent
    2. descend so the jaw tips are at table level, still behind the cube
    3. slide forward so the cube seats in the jaw pocket
    4. ramp the gripper closed slowly (a fast close smacks the cube away)
    5. lift; retry from step 1 at the cube's new position if the grasp slipped
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from so101_oink.ee_target import WORKSPACE_HIGH, WORKSPACE_LOW, EETarget
from so101_oink.env import CUBE_HALF, GRIPPER_OPEN, SO101PickCubeEnv

if TYPE_CHECKING:
    from so101_oink.oink_controller import OinkController

GRASP_PITCH = 0.6  # rad beyond home pitch: near-vertical approach
POCKET_PERP = 0.01  # jaw pocket offset from EE frame toward the moving jaw
BEHIND_OFFSET = 0.008  # descend clearance behind the cube
SEAT_OFFSET = 0.003  # remaining clearance once the cube is seated
CLOSE_TICKS = 22  # gripper close ramp duration (policy ticks)
HOVER_HEIGHT = 0.09
GRASP_DEPTH = -0.012  # EE frame height relative to the cube center when the tips touch the table
LIFT_HEIGHT = 0.13

Command = tuple[np.ndarray, float]  # (world_T_gripper_frame, gripper_rad)


def _moving_jaw_dir(ee: EETarget) -> np.ndarray:
    """Unit vector from the fixed jaw toward the moving jaw (rolls with the wrist)."""
    return -(ee.target_rot @ np.array([1.0, 0.0, 0.0]))


def _pocket_offset(ee: EETarget) -> np.ndarray:
    return POCKET_PERP * _moving_jaw_dir(ee)


def _horiz(ee: EETarget) -> np.ndarray:
    """Horizontal (unit) projection of the fixed-jaw-to-moving-jaw direction."""
    h = _moving_jaw_dir(ee)
    h[2] = 0.0
    return h / np.linalg.norm(h)


def cube_relative_roll(env: SO101PickCubeEnv) -> float:
    """Roll command that aligns the jaw plane with the cube's nearest face pair.

    The cube has 4-fold symmetry, so the relative yaw is wrapped to [-45, 45]
    degrees; world jaw yaw responds as -roll (the tool axis points down).
    """
    adr = env.cube_qpos_adr
    qw, qx, qy, qz = env.data.qpos[adr + 3 : adr + 7]
    cube_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cube = env.cube_pos
    approach_yaw = np.arctan2(cube[1], cube[0])
    rel = cube_yaw - approach_yaw
    rel = (rel + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(-rel)


def _goto(
    ctl: "OinkController",
    ee: EETarget,
    gripper: float,
    pocket_goal: np.ndarray,
    pitch_goal: float = GRASP_PITCH,
    roll_goal: float = 0.0,
    max_ticks: int = 250,
    step: float = 0.005,
) -> Iterator[Command]:
    """Walk the EE target toward ``pocket_goal`` and wait for the solver to settle on it."""
    best_err = np.inf
    stalled = 0
    for _ in range(max_ticks):
        dpitch = np.clip(pitch_goal - ee.pitch, -0.02, 0.02)
        droll = np.clip(roll_goal - ee.roll, -0.03, 0.03)
        goal_ee = np.clip(pocket_goal - _pocket_offset(ee), WORKSPACE_LOW, WORKSPACE_HIGH)
        delta = goal_ee - ee.target_pos
        dist = np.linalg.norm(delta)
        ee.move(delta / max(dist, 1e-9) * min(step, dist), dpitch, droll)
        yield ee.pose(), gripper
        if dist < 1e-4 and abs(pitch_goal - ee.pitch) < 1e-3 and abs(roll_goal - ee.roll) < 1e-3:
            # settle until the commanded EE converges, or stops improving (the
            # 5-DOF arm cannot reach every orientation exactly, which must not
            # stall the phase for its whole tick budget)
            err = float(np.linalg.norm(ctl.fk(ctl.q_full[ctl.q_idx])[:3, 3] - ee.target_pos))
            if err < 0.004:
                return
            if err < best_err - 3e-4:
                best_err = err
                stalled = 0
            else:
                stalled += 1
                if stalled >= 8:
                    return


def scripted_pick(env: SO101PickCubeEnv, ctl: "OinkController", ee: EETarget, retries: int = 2) -> Iterator[Command]:
    """Yield (pose, gripper) commands that pick up the target cube; stops after success or gives up."""
    close_target = float(env.action_space.low[5]) + 0.075  # slight squeeze margin
    for _ in range(1 + retries):
        cube = env.cube_pos.copy()
        cube[2] = CUBE_HALF
        roll = cube_relative_roll(env)  # align jaw plane with the cube's faces
        yield from _goto(
            ctl,
            ee,
            GRIPPER_OPEN,
            cube + [0, 0, HOVER_HEIGHT] - BEHIND_OFFSET * _horiz(ee),
            roll_goal=roll,
        )
        yield from _goto(
            ctl,
            ee,
            GRIPPER_OPEN,
            cube + [0, 0, GRASP_DEPTH] - BEHIND_OFFSET * _horiz(ee),
            roll_goal=roll,
        )
        yield from _goto(
            ctl,
            ee,
            GRIPPER_OPEN,
            cube + [0, 0, GRASP_DEPTH] - SEAT_OFFSET * _horiz(ee),
            roll_goal=roll,
            step=0.002,
        )
        for i in range(CLOSE_TICKS):
            yield (
                ee.pose(),
                GRIPPER_OPEN + (close_target - GRIPPER_OPEN) * (i + 1) / CLOSE_TICKS,
            )
        yield from _goto(
            ctl,
            ee,
            close_target,
            cube + [0, 0, LIFT_HEIGHT],
            roll_goal=roll,
            max_ticks=120,
            step=0.004,
        )
        for _ in range(10):
            yield ee.pose(), close_target
        if env._is_success():
            return

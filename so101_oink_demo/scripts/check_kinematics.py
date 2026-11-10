"""Sanity checks for the RoboPlan <-> MuJoCo pairing (run this first):  pixi run so101-check

1. RoboPlan's FK of the RoboPlan SO-101 URDF matches the MuJoCo gripper frame.
2. Both OInK loadouts converge, each in ~30 us per step.
3. The table barrier stops the gripper frame at z = table_z even when the
   target is far below the table.
"""

import time

import mujoco
import numpy as np
from so101_oink.ee_target import EETarget
from so101_oink.env import SO101PickCubeEnv
from so101_oink.oink_controller import OinkConfig, OinkController


def track(ctl: OinkController, steps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(steps):
        ctl.step()
    return (time.perf_counter() - t0) / steps * 1e6


def main() -> None:
    env = SO101PickCubeEnv()
    env.reset(seed=0)
    home = env.home_joint_pos
    ctl = OinkController("cartesian")

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(env.action_space.low, env.action_space.high)
        env.data.qpos[env.qpos_ids] = q
        mujoco.mj_forward(env.model, env.data)
        worst = max(worst, np.linalg.norm(env.ee_pos - ctl.fk(q)[:3, 3]))
    print(f"FK position mismatch over 200 random configurations: {worst * 1e3:.3f} mm")
    assert worst < 1e-3

    # Cartesian loadout: a yaw-coupled target toward a cube, pitched down.
    ctl.reset(home)
    ee = EETarget()
    ee.reset(ctl.fk(home))
    ee.move(np.array([0.0, -0.08, -0.05]), delta_pitch=0.6)
    ctl.set_pose_target(ee.pose())
    us = track(ctl, 150)
    err = np.linalg.norm(ctl.fk(ctl.q_full[ctl.q_idx])[:3, 3] - ee.target_pos)
    print(f"cartesian loadout: {us:.0f} us/step, position error {err * 1e3:.2f} mm, barriers {ctl.barrier_report()}")
    assert err < 0.005

    ctl = OinkController("joint")
    ctl.reset(home)
    target = home[:5] + [0.3, -0.2, 0.2, 0.1, 0.5]
    ctl.set_joint_target(target)
    us = track(ctl, 100)
    print(f"joint loadout: {us:.0f} us/step, max joint error {np.degrees(np.abs(ctl.q_arm - target).max()):.3f} deg")
    assert np.abs(ctl.q_arm - target).max() < np.radians(0.5)

    for table_barrier in (True, False):
        ctl = OinkController("cartesian", OinkConfig(table_barrier=table_barrier))
        ctl.reset(home)
        target = ctl.fk(home)
        target[2, 3] = -0.10
        ctl.set_pose_target(target)
        track(ctl, 300)
        z = ctl.fk(ctl.q_full[ctl.q_idx])[2, 3]
        print(
            f"target z = -100 mm, table barrier {'on ' if table_barrier else 'off'}: gripper frame at z = {z * 1e3:+.1f} mm"
        )
        assert (z > -0.002) if table_barrier else (z < -0.05)

    env.close()
    print("all checks passed")


if __name__ == "__main__":
    main()

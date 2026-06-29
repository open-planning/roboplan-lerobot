"""Demo: move the Kinova EE vertically and hold.

Usage (inside the Docker container):
    python scripts/demo.py
    python scripts/demo.py --delta-z 0.15 --steps 200

Open http://localhost:8080 in your browser to watch live.
Run with --help to see all options.
"""

from __future__ import annotations

import time
from typing import Literal

import gymnasium as gym
import tyro

import roboplan_lerobot  # noqa: F401 -- registers the gym environment
from roboplan_lerobot.configs.kinova import KinovaEnvConfig


def main(
    delta_z: float = 0.1,
    steps: int = 150,
    render_mode: Literal["viser", "rgb_array", "none"] = "viser",
) -> None:
    """Move the Kinova EE delta_z metres upward, hold, then exit.

    Args:
        delta_z: How far to move the end-effector vertically (metres).
        steps: Total number of control steps.
        render_mode: viser for live 3D view, rgb_array for offscreen, none for headless.
    """
    config = KinovaEnvConfig()
    effective_render_mode: str | None = None if render_mode == "none" else render_mode

    env = gym.make(
        "roboplan_lerobot/KinovaReach-v0",
        config=config,
        render_mode=effective_render_mode,
    )
    try:
        obs, _ = env.reset()
        print("home EE pos:", obs[:3])
        if effective_render_mode == "viser":
            print(f"open http://localhost:{config.viser_port} in your browser")

        # obs[:7] is [pos(3), quat_xyzw(4)]; shift z up by delta_z
        target = obs[:7].copy()
        target[2] += delta_z

        for _ in range(steps):
            obs, _, _, _, info = env.step(target)
            if info.get("ik_failed"):
                print("warning: IK solver failed this step")
            time.sleep(env.unwrapped.dt)

        print("final EE pos:", obs[:3])

        if effective_render_mode == "viser":
            print("press Ctrl+C to exit")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

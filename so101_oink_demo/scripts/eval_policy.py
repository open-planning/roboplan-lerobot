"""Roll out a trained policy in the sim with OInK executing its actions; report success rate.

    pixi run so101-eval --policy-path outputs/train/so101_diffusion_joint/checkpoints/last/pretrained_model --episodes 20
    pixi run so101-eval --policy-path ... --show                       # watch (--viewer viser for a browser)
    pixi run so101-eval --policy-path ... --show --oink.table_z=0.02   # change the barriers at rollout time
    pixi run so101-eval --policy-path ... --num-inference-steps 10     # diffusion: 10 DDIM steps, ~10x faster

The policy's action space is read from the checkpoint (6 outputs = joint,
10 = Cartesian) and the robot gets the matching OInK loadout. Each action
becomes a task target for one 25 Hz tick; the solver tracks it under the joint
limits and barriers, so unsafe outputs are filtered rather than obeyed.
"""

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot_robot_so101_oink import SO101Oink, SO101OinkConfig
from sim_pipelines import FPS, add_common_args, robot_kwargs
from so101_oink.env import JOINT_NAMES, TASKS
from so101_oink.spaces import ACTION_KEYS, POSE_KEYS

STATE_KEYS = [f"{j}.pos" for j in JOINT_NAMES] + POSE_KEYS


def to_batch(obs: dict, device: str, task: str) -> dict:
    batch = {"observation.state": torch.tensor([[obs[k] for k in STATE_KEYS]], device=device), "task": [task]}
    for cam in ("wrist", "front"):
        batch[f"observation.images.{cam}"] = (
            torch.from_numpy(obs[cam]).permute(2, 0, 1)[None].float().div(255).to(device)
        )
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="checkpoint pretrained_model dir")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-ticks", type=int, default=250, help="sim ticks per episode (25/s)")
    parser.add_argument("--show", action="store_true", help="display in the viewer, paced to real time")
    parser.add_argument("--task", default=None, help="override the language instruction")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="diffusion: sample with DDIM in this many steps instead of the trained DDPM schedule",
    )
    add_common_args(parser, action_space=False)
    args = parser.parse_args()
    if not Path(args.policy_path).is_dir():
        parser.error(f"--policy-path is not a local directory: {args.policy_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    if args.num_inference_steps is not None:
        if cfg.type != "diffusion":
            parser.error("--num-inference-steps only applies to diffusion policies")
        # DDIM reuses the network trained for DDPM's noise schedule, just with larger, deterministic steps.
        cfg.noise_scheduler_type = "DDIM"
        cfg.num_inference_steps = args.num_inference_steps
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=args.policy_path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    action_dim = cfg.output_features["action"].shape[0]
    action_space = next(space for space, keys in ACTION_KEYS.items() if len(keys) == action_dim)
    keys = ACTION_KEYS[action_space]
    print(f"policy: {cfg.type}, action space: {action_space}, device: {device}")

    robot = SO101Oink(
        SO101OinkConfig(show_viewer=args.show, seed=args.seed, action_space=action_space, **robot_kwargs(args))
    )
    robot.connect()

    # Ctrl+C finishes the current tick and then wraps up normally. An exception would leave the
    # passive MuJoCo viewer thread running, which deadlocks glfw's exit hook; pixi also forwards
    # the signal, so a second one could land in the middle of the cleanup.
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    successes = wrong = done = 0
    lowest = np.inf
    for ep in range(args.episodes):
        color = args.colors[ep % len(args.colors)]
        robot.reset_scene(seed=args.seed + ep, target_color=color)
        policy.reset()
        wrong_cube = False
        for tick in range(1, args.max_ticks + 1):
            if stop.is_set():
                break
            t0 = time.perf_counter()
            batch = preprocessor(to_batch(robot.get_observation(), device, args.task or TASKS[color]))
            with torch.inference_mode():
                action = postprocessor(policy.select_action(batch))[0].cpu().numpy()
            robot.send_action(dict(zip(keys, action.astype(float))))
            lowest = min(lowest, robot.controller.barrier_report().get("gripper_frame_link_above_table", np.inf))
            wrong_cube |= robot.wrong_cube_lifted
            if robot.is_success:
                break
            if args.show:
                time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
        if stop.is_set():
            print(f"\ninterrupted after {done} episodes")
            break
        done += 1
        successes += robot.is_success
        wrong += wrong_cube and not robot.is_success
        outcome = "success" if robot.is_success else ("failure (wrong cube)" if wrong_cube else "failure")
        print(f"episode {ep + 1}/{args.episodes} [{color}]: {outcome} after {tick} ticks")

    if done:
        print(f"\nsuccess rate: {successes}/{done} ({100 * successes / done:.0f}%), wrong cube: {wrong}")
        print(f"lowest gripper-frame height above the table barrier: {lowest * 1e3:+.1f} mm, "
              f"solver failures: {robot.controller.solver_failures}")  # fmt: skip
    robot.disconnect()
    # Skip interpreter teardown: torch and the roboplan bindings unload in an order that
    # makes nanobind print a spurious "leaked instances" report on a normal exit.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

"""Collect pick-cube episodes with the scripted expert (headless).

    pixi run so101-collect --repo-id you/so101_pick_joint --root data/pick_joint --episodes 100
    pixi run so101-collect --repo-id you/so101_pick_cart --root data/pick_cart --episodes 100 --action-space cartesian
    pixi run so101-collect --repo-id you/so101_pick_two --root data/pick_two --episodes 200 --colors red green

The expert commands gripper-frame poses; OInK turns them into motion and the
dataset is recorded in the chosen action space, so one expert produces both
kinds of dataset. Failed attempts are discarded. --show-viewer watches it live.

--perturb P slides the cube a few centimetres away while the gripper descends in
a fraction P of the episodes, so the expert's retry is recorded and the policy
learns to re-grasp instead of hovering after a miss.
"""

import argparse
import time

import numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)
from lerobot_robot_so101_oink import SO101Oink, SO101OinkConfig
from sim_pipelines import (
    FPS,
    add_common_args,
    oink_config,
    pose_action_pipeline,
    robot_kwargs,
)
from so101_oink.ee_target import EETarget
from so101_oink.env import TASKS
from so101_oink.scripted import scripted_pick
from so101_oink.spaces import GRIPPER_KEY, pose_to_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", action="store_true", help="watch in the viewer, paced to real time")
    parser.add_argument("--perturb", type=float, default=0.0, help="fraction of episodes with a mid-descent cube nudge")
    add_common_args(parser)
    args = parser.parse_args()

    robot = SO101Oink(SO101OinkConfig(show_viewer=args.show_viewer, **robot_kwargs(args)))
    robot.connect()
    to_action = pose_action_pipeline(args.action_space, oink_config(args))

    features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=True),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
    )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, root=args.root, features=features, robot_type=robot.name, image_writer_threads=8
    )

    ee = EETarget()
    rng = np.random.default_rng(args.seed)
    saved = 0
    try:
        while saved < args.episodes:
            color = args.colors[saved % len(args.colors)]
            robot.reset_scene(seed=int(rng.integers(2**31)), target_color=color)
            ee.reset(robot.ee_pose)
            to_action.reset()
            nudge = rng.random() < args.perturb
            for pose, gripper in scripted_pick(robot.env, robot.controller, ee):
                t0 = time.perf_counter()
                if nudge and robot.env.ee_pos[2] < 0.04 and gripper > 1.0:  # descending, gripper still open
                    angle = rng.uniform(0, 2 * np.pi)
                    delta = rng.uniform(0.02, 0.04) * np.array([np.cos(angle), np.sin(angle)])
                    other = robot.env.cube_pos_of("green" if color == "red" else "red")
                    if np.linalg.norm(robot.env.cube_pos[:2] + delta - other[:2]) < 0.05:
                        delta = -delta
                    robot.env.nudge_cube(delta)
                    nudge = False
                obs = robot.get_observation()
                action = to_action(({**pose_to_dict(pose), GRIPPER_KEY: float(np.degrees(gripper))}, obs))
                dataset.add_frame({
                    **build_dataset_frame(features, obs, prefix=OBS_STR),
                    **build_dataset_frame(features, action, prefix=ACTION),
                    "task": TASKS[color],
                })  # fmt: skip
                robot.send_action(action)
                if args.show_viewer:
                    time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
                if robot.is_success:
                    break
            if robot.is_success:
                dataset.save_episode()
                saved += 1
                print(f"episode {saved}/{args.episodes} saved: {color} cube")
            else:
                dataset.clear_episode_buffer()
                print("pick failed, episode discarded")
    finally:
        robot.disconnect()
        dataset.finalize()  # an unfinalized dataset cannot be loaded back
    print(f"Done: {saved} episodes -> {dataset.root}")


if __name__ == "__main__":
    main()

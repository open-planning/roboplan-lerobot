"""Record a LeRobotDataset by teleoperating the SO-101 through OInK.

    pixi run so101-record --repo-id you/so101_pick_joint --root data/pick_joint --episodes 50 --display-data
    pixi run so101-record --repo-id you/so101_pick_cart --root data/pick_cart --episodes 50 --action-space cartesian
    pixi run so101-record --repo-id you/so101_pick_two --root data/pick_two --episodes 100 --colors red green

--action-space picks what the dataset stores and which OInK task the robot tracks.
The keyboard always drives the end effector (arrows/w/s = move, q/e = pitch,
a/d = roll, space = gripper). Episode controls: n = save, r = re-record,
ESC = stop and finalize. Uses lerobot's own record_loop.
"""

import argparse
import logging

from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot_robot_so101_oink import SO101Oink, SO101OinkConfig
from lerobot_teleoperator_keyboard_pose import KeyboardPose, KeyboardPoseConfig
from lerobot_teleoperator_keyboard_pose.keys import ESC, make_listener
from sim_pipelines import (
    FPS,
    action_pipeline,
    add_common_args,
    identity_observation_pipeline,
    key_source,
    oink_config,
    robot_kwargs,
    start_browser_keys,
    teleop_action_pipeline,
)
from so101_oink.env import TASKS

EPISODE_TIME_S = 120


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="dataset repo id, e.g. you/so101_pick_joint")
    parser.add_argument("--root", default=None, help="local dataset root (default: HF cache)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--display-data", action="store_true", help="live rerun view")
    add_common_args(parser)
    args = parser.parse_args()

    # record_loop warns on every tick that misses the target fps; once is enough.
    logging.getLogger().addFilter(lambda r: not r.getMessage().startswith("Record loop is running slower"))

    robot = SO101Oink(SO101OinkConfig(show_viewer=True, **robot_kwargs(args)))
    robot.connect()
    source = key_source(args)
    browser_port = start_browser_keys(robot) if source == "browser" else 0
    teleop = KeyboardPose(KeyboardPoseConfig(key_source=source, browser_port=browser_port))
    teleop_processor = teleop_action_pipeline(args.action_space, oink_config(args))
    robot_processor = action_pipeline([])
    observation_processor = identity_observation_pipeline()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        root=args.root,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                teleop_processor, create_initial_features(action=teleop.action_features), use_videos=True
            ),
            aggregate_pipeline_dataset_features(
                observation_processor, create_initial_features(observation=robot.observation_features), use_videos=True
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=8,
    )
    teleop.connect()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def on_press(key: str):
        if key == "n":
            events["exit_early"] = True
        elif key == "r":
            events["rerecord_episode"] = events["exit_early"] = True
        elif key == ESC:
            events["stop_recording"] = events["exit_early"] = True

    listener = make_listener(source, on_press, browser_port=browser_port)
    listener.start()

    if args.display_data:
        from lerobot.utils.rerun_visualization import init_rerun

        init_rerun(session_name="so101_oink_record")

    print("arrows/w/s = move, q/e = pitch, a/d = roll, space = gripper | n = save, r = re-record, ESC = stop (type here)")
    episode = 0
    try:
        while episode < args.episodes and not events["stop_recording"]:
            color = args.colors[episode % len(args.colors)]
            robot.reset_scene(target_color=color)
            teleop.reset_state()
            teleop_processor.reset()
            print(f"recording episode {episode + 1}/{args.episodes} - lift the {color.upper()} cube!")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_processor,
                robot_action_processor=robot_processor,
                robot_observation_processor=observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_S,
                single_task=TASKS[color],
                display_data=args.display_data,
            )
            if events["rerecord_episode"]:
                events["rerecord_episode"] = events["exit_early"] = False
                dataset.clear_episode_buffer()
                print("re-recording episode")
            elif not events["stop_recording"]:
                dataset.save_episode()
                episode += 1
    finally:
        robot.disconnect()
        teleop.disconnect()
        listener.stop()
        dataset.finalize()
    print(f"Done: {episode} episodes -> {dataset.root}")


if __name__ == "__main__":
    main()

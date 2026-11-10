"""Keyboard-teleoperate the SO-101 in MuJoCo through OInK (no recording).

    pixi run so101-teleoperate
    pixi run so101-teleoperate --action-space cartesian   # the robot tracks poses with a FrameTask
    pixi run so101-teleoperate --viewer viser              # browser viewer, e.g. on a remote machine
    pixi run so101-teleoperate --oink.table_z=0.02         # raise the table barrier 2 cm

arrows = EE forward/back/left/right, w/s = up/down, q/e = pitch,
a/d = roll jaws, space = toggle gripper, x = reset scene, ESC = quit.
The barrier values are printed once a second: push into the table or fold the
arm onto itself and watch them reach zero as the arm stops.
"""

import argparse
import time

from lerobot_robot_so101_oink import SO101Oink, SO101OinkConfig
from lerobot_teleoperator_keyboard_pose import KeyboardPose, KeyboardPoseConfig
from lerobot_teleoperator_keyboard_pose.keys import ESC, make_listener
from sim_pipelines import (
    FPS,
    add_common_args,
    key_source,
    oink_config,
    robot_kwargs,
    start_browser_keys,
    teleop_action_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()

    robot = SO101Oink(SO101OinkConfig(show_viewer=True, target_color=args.colors[0], **robot_kwargs(args)))
    robot.connect()
    source = key_source(args)
    browser_port = start_browser_keys(robot) if source == "browser" else 0
    teleop = KeyboardPose(KeyboardPoseConfig(key_source=source, browser_port=browser_port))
    pipeline = teleop_action_pipeline(args.action_space, oink_config(args))

    events = {"reset": False, "quit": False}

    def on_press(key: str):
        if key == ESC:
            events["quit"] = True
        elif key == "x":
            events["reset"] = True

    listener = make_listener(source, on_press, browser_port=browser_port)
    listener.start()
    teleop.connect()
    print("arrows/w/s = move, q/e = pitch, a/d = roll, space = gripper, x = reset, ESC = quit (type here)")
    print(f"action space: {args.action_space}; target: the {robot.target_color.upper()} cube")

    resets = 0
    was_success = False
    last_report = 0.0
    try:
        while not events["quit"] and robot.viewer.is_running():
            t0 = time.perf_counter()
            if events["reset"]:
                events["reset"] = False
                resets += 1
                robot.reset_scene(target_color=args.colors[resets % len(args.colors)])
                teleop.reset_state()
                pipeline.reset()
                was_success = False
                print(f"\ntarget: the {robot.target_color.upper()} cube")

            robot.send_action(pipeline((teleop.get_action(), robot.get_observation())))

            if robot.is_success and not was_success:
                print(f"\nSUCCESS - {robot.target_color} cube lifted! (x to reset)")
            was_success = robot.is_success
            if t0 - last_report > 1.0:
                last_report = t0
                report = "  ".join(f"{k}={v * 1e3:.0f}mm" for k, v in robot.controller.barrier_report().items())
                print(f"\rbarriers: {report}   ", end="", flush=True)

            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    finally:
        robot.disconnect()
        teleop.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()

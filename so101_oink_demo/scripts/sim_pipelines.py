"""Shared arguments and processor pipelines for the demo scripts.

The teleop action pipeline turns keyboard offsets into the dataset's action
space with OInK doing the kinematics (see lerobot_robot_so101_oink/processors.py):

    keyboard offsets -> EETargetFromOffsets -> pose + gripper.pos          (cartesian)
                                            -> OinkPoseToJoints -> {joint}.pos   (joint)
"""

from dataclasses import fields

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot_robot_so101_oink.config_so101_oink import VIEWERS
from lerobot_robot_so101_oink.processors import EETargetFromOffsets, OinkPoseToJoints
from so101_oink.env import CUBE_COLORS
from so101_oink.oink_controller import OinkConfig

FPS = 25


def add_common_args(parser, action_space=True):
    """--colors, --action-space, --viewer/--viser-port, and --oink.<field> for every OinkConfig field."""
    parser.add_argument(
        "--colors",
        nargs="+",
        choices=CUBE_COLORS,
        default=["red"],
        metavar="COLOR",
        help="target cube color(s); episodes alternate through them (default: red)",
    )
    if action_space:
        parser.add_argument(
            "--action-space",
            choices=["joint", "cartesian"],
            default="joint",
            help="dataset action space and OInK loadout (default: joint)",
        )
    parser.add_argument(
        "--viewer",
        choices=VIEWERS,
        default="native",
        help="native = MuJoCo window; viser = browser viewer, works on a remote machine",
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="port for --viewer viser (default: 8080)")
    oink = parser.add_argument_group("OInK solver (so101_oink/oink_controller.py)")
    for f in fields(OinkConfig):
        to_type = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
        oink.add_argument(
            f"--oink.{f.name}", dest=f"oink_{f.name}", type=to_type, default=f.default, metavar=f.type.__name__.upper()
        )


def oink_config(args) -> OinkConfig:
    return OinkConfig(**{f.name: getattr(args, f"oink_{f.name}") for f in fields(OinkConfig)})


def robot_kwargs(args) -> dict:
    """SO101OinkConfig fields set by add_common_args."""
    kwargs = {"oink": oink_config(args), "viewer": args.viewer, "viser_port": args.viser_port}
    if hasattr(args, "action_space"):
        kwargs["action_space"] = args.action_space
    return kwargs


def key_source(args) -> str:
    """Keys come from the terminal, or from the browser tab with the web viewer."""
    return "browser" if args.viewer == "viser" else "terminal"


def start_browser_keys(robot) -> int:
    """Serve key presses from the tab of ``robot``'s viser viewer; returns the websocket port."""
    from lerobot_teleoperator_keyboard_pose.browser_keys import BrowserKeys

    keys = BrowserKeys.get(robot.viewer.server.get_port() + 1)
    keys.attach(robot.viewer.server)
    print(f"keys are read from the browser tab (websocket on port {keys.port})")
    return keys.port


def action_pipeline(steps) -> RobotProcessorPipeline:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=steps, to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action
    )


def teleop_action_pipeline(action_space: str, oink: OinkConfig) -> RobotProcessorPipeline:
    """Keyboard offsets -> action in ``action_space``."""
    steps = [EETargetFromOffsets()] + ([OinkPoseToJoints(oink, FPS)] if action_space == "joint" else [])
    return action_pipeline(steps)


def pose_action_pipeline(action_space: str, oink: OinkConfig) -> RobotProcessorPipeline:
    """Pose + gripper.pos (what the scripted expert emits) -> action in ``action_space``."""
    return action_pipeline([OinkPoseToJoints(oink, FPS)] if action_space == "joint" else [])


def identity_observation_pipeline() -> RobotProcessorPipeline:
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[], to_transition=observation_to_transition, to_output=transition_to_observation
    )

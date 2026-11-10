"""Keyboard end-effector pose teleoperator.

    arrows      EE forward/back and left/right      w / s   EE up / down
    q / e       pitch the gripper                   a / d   roll the jaws
    space       toggle gripper open/close

Episode keys (n/r/x/ESC) are handled by the scripts, not here.
"""

import threading
import time
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_keyboard_pose import KeyboardPoseConfig
from .keys import DOWN, LEFT, RIGHT, SPACE, UP, make_listener

MOVE_KEYS = {
    UP: np.array([1.0, 0.0, 0.0]),
    DOWN: np.array([-1.0, 0.0, 0.0]),
    LEFT: np.array([0.0, 1.0, 0.0]),
    RIGHT: np.array([0.0, -1.0, 0.0]),
    "w": np.array([0.0, 0.0, 1.0]),
    "s": np.array([0.0, 0.0, -1.0]),
}
PITCH_KEYS = {"q": 1.0, "e": -1.0}
ROLL_KEYS = {"a": 1.0, "d": -1.0}
PITCH_RANGE = (-1.0, 1.0)
ROLL_RANGE = (-0.9, 0.9)

ACTION_KEYS = [
    "enabled",
    "delta_x",
    "delta_y",
    "delta_z",
    "pitch",
    "roll",
    "gripper_vel",
]


class KeyboardPose(Teleoperator):
    config_class = KeyboardPoseConfig
    name = "keyboard_pose"

    def __init__(self, config: KeyboardPoseConfig):
        super().__init__(config)
        self.config = config
        self._held: set = set()
        self._lock = threading.Lock()
        self._listener = None  # Subscription on the terminal or browser key source
        self._last_t: float | None = None
        self.gripper_open: bool | None = None  # untouched until the first space press
        self._offset = np.zeros(3)
        self._pitch = 0.0
        self._roll = 0.0
        self._rates = np.zeros(5)  # filtered xyz / pitch / roll rates

    # ------------------------------------------------------------- features

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    # ----------------------------------------------------------- connection

    @property
    def is_connected(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def connect(self, calibrate: bool = True) -> None:
        self._listener = make_listener(
            self.config.key_source,
            self._on_press,
            self._on_release,
            self.config.browser_port,
        )
        self._listener.start()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------- keyboard

    def _on_press(self, key: str):
        with self._lock:
            if key == SPACE and key not in self._held:
                self.gripper_open = not self.gripper_open  # from the untouched state, the first press opens
            self._held.add(key)

    def _on_release(self, key: str):
        with self._lock:
            self._held.discard(key)

    # --------------------------------------------------------------- action

    def reset_state(self) -> None:
        """Zero the accumulated offsets (call together with the pipeline reset)."""
        self._offset = np.zeros(3)
        self._pitch = 0.0
        self._roll = 0.0
        self._rates = np.zeros(5)
        self.gripper_open = None
        self._last_t = None

    def get_action(self) -> dict[str, Any]:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now

        with self._lock:
            held = set(self._held)
            gripper_open = self.gripper_open

        move = sum((d for k, d in MOVE_KEYS.items() if k in held), np.zeros(3))
        if np.any(move):
            move *= self.config.move_speed / np.linalg.norm(move)
        pitch_dir = sum(v for k, v in PITCH_KEYS.items() if k in held)
        roll_dir = sum(v for k, v in ROLL_KEYS.items() if k in held)
        target = np.array([*move, pitch_dir * self.config.pitch_speed, roll_dir * self.config.pitch_speed])

        # first-order filter: keys ramp the rates instead of switching them, so taps and
        # autorepeat gaps become smooth motion
        alpha = 1.0 if self.config.smoothing <= 0 else min(1.0, dt / self.config.smoothing)
        self._rates += (target - self._rates) * alpha
        self._offset += self._rates[:3] * dt
        self._pitch = float(np.clip(self._pitch + self._rates[3] * dt, *PITCH_RANGE))
        self._roll = float(np.clip(self._roll + self._rates[4] * dt, *ROLL_RANGE))

        return {
            "enabled": 1.0,
            "delta_x": float(self._offset[0]),
            "delta_y": float(self._offset[1]),
            "delta_z": float(self._offset[2]),
            "pitch": self._pitch,
            "roll": self._roll,
            "gripper_vel": 0.0 if gripper_open is None else np.copysign(self.config.gripper_vel, gripper_open - 0.5),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

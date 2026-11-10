"""Key presses read from the terminal that launched the script.

The terminal is switched to cbreak mode (no echo, no line buffering) and read
on a thread. Terminals report presses but not releases, so every key event
holds its key for ``HOLD`` seconds: a tap moves one step, and the keyboard's
autorepeat keeps a pressed key held until it is let go.

One ``TerminalKeys`` per terminal is shared by everyone who calls ``get()``
(the teleoperator and the script's own episode keys), since stdin can only be
read once. ``listener()`` has the same surface as the browser source.
"""

import atexit
import os
import select
import sys
import termios
import threading
import time
import tty

from .keys import Subscription

HOLD = 0.12  # s a key stays held after an event; autorepeat (~30 Hz) renews it
_ESCAPES = {"A": "up", "B": "down", "C": "right", "D": "left"}  # final byte of ESC [ x
_SPECIAL = {" ": "space", "\r": "enter", "\n": "enter", "\x1b": "esc"}


class TerminalKeys:
    _instances: dict[int, "TerminalKeys"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, fd: int | None = None) -> "TerminalKeys":
        fd = sys.stdin.fileno() if fd is None else fd
        with cls._lock:
            if fd not in cls._instances:
                cls._instances[fd] = cls(fd)
            return cls._instances[fd]

    def __init__(self, fd: int):
        if not os.isatty(fd):
            raise RuntimeError("keyboard teleop needs a terminal on stdin (or use --viewer viser for browser keys)")
        self._fd = fd
        self._subs: list[Subscription] = []
        self._saved = None
        self._thread: threading.Thread | None = None

    def listener(self, on_press=None, on_release=None) -> Subscription:
        return Subscription(self, on_press, on_release)

    # ------------------------------------------------------------ internals

    def _subscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.append(sub)
            if self._thread is None:
                self._saved = termios.tcgetattr(self._fd)
                atexit.register(termios.tcsetattr, self._fd, termios.TCSADRAIN, self._saved)  # also after a crash
                tty.setcbreak(self._fd)
                self._thread = threading.Thread(target=self._run, daemon=True, name="terminal-keys")
                self._thread.start()

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
            if not self._subs and self._thread is not None:
                self._thread = None  # the loop exits on its next pass
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def _fire(self, pressed: bool, key: str) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            cb = sub.on_press if pressed else sub.on_release
            if cb is not None:
                cb(key)

    def _read_keys(self) -> list[str]:
        """Everything waiting on stdin (autorepeat batches several events), as canonical names."""
        buf = os.read(self._fd, 256)
        if buf.endswith((b"\x1b", b"\x1b[")) and select.select([self._fd], [], [], 0.05)[0]:
            buf += os.read(self._fd, 256)  # finish a split escape sequence (a lone ESC has nothing more)
        keys, i = [], 0
        while i < len(buf):
            if buf[i : i + 2] == b"\x1b[" and i + 2 < len(buf):
                keys.append(_ESCAPES.get(buf[i + 2 : i + 3].decode(errors="ignore")))
                i += 3
            else:
                ch = buf[i : i + 1].decode(errors="ignore")
                keys.append(_SPECIAL.get(ch, ch.lower() if ch.isprintable() else None))
                i += 1
        return [k for k in keys if k is not None]

    def _run(self) -> None:
        me = threading.current_thread()
        expiry: dict[str, float] = {}
        while self._thread is me:
            if select.select([self._fd], [], [], 0.02)[0]:
                for key in self._read_keys():
                    if key not in expiry:
                        self._fire(True, key)
                    expiry[key] = time.monotonic() + HOLD
            now = time.monotonic()
            for key in [k for k, t in expiry.items() if t < now]:
                del expiry[key]
                self._fire(False, key)
        for key in expiry:
            self._fire(False, key)

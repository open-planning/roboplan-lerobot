"""Key names and listener factory shared by the two key sources.

Keys are canonical strings: single lowercase characters ("w", "x") plus
"up", "down", "left", "right", "space", "enter" and "esc".

``make_listener`` returns a ``Subscription`` (``start`` / ``stop`` /
``is_alive``) whose callbacks receive canonical names, from either the
terminal that launched the script or the viser viewer's browser tab.
"""

from collections.abc import Callable

UP, DOWN, LEFT, RIGHT, SPACE, ESC = "up", "down", "left", "right", "space", "esc"


class Subscription:
    """One listener on a shared key source."""

    def __init__(self, source, on_press, on_release):
        self._source = source
        self.on_press = on_press
        self.on_release = on_release
        self._alive = False

    def start(self) -> None:
        self._source._subscribe(self)
        self._alive = True

    def stop(self) -> None:
        self._source._unsubscribe(self)
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


def make_listener(
    source: str,
    on_press: Callable[[str], None] | None = None,
    on_release: Callable[[str], None] | None = None,
    browser_port: int = 8081,
) -> Subscription:
    if source == "browser":
        from .browser_keys import BrowserKeys

        return BrowserKeys.get(browser_port).listener(on_press=on_press, on_release=on_release)
    if source == "terminal":
        from .terminal_keys import TerminalKeys

        return TerminalKeys.get().listener(on_press=on_press, on_release=on_release)
    raise ValueError(f"unknown key source {source!r}, expected 'terminal' or 'browser'")

"""Key presses captured in the browser tab of a viser viewer.

viser has no keyboard events of its own, so a small snippet injected into the
viewer's GUI panel (``attach``) listens for keydown/keyup in the page and
forwards them over a second websocket to this process. ``listener()`` delivers
canonical key names (see keys.py), so the teleop code is the same for both
sources.

One ``BrowserKeys`` server per port is shared by everyone who calls ``get``
(the teleoperator and the script's own episode keys), so a page needs one
connection. Like viser, the server moves up to the next free port if the
requested one is taken (several users on one machine); ``port`` is the one it
actually listens on.
"""

import json
import threading
from collections.abc import Callable

from websockets.sync.server import serve

from .keys import Subscription

# browser KeyboardEvent.key (lowercased) -> canonical name
_SPECIAL = {
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    " ": "space",
    "escape": "esc",
    "enter": "enter",
    "shift": "shift",
}

# Which keys the page intercepts (and keeps from viser's fly camera, which
# also uses w/a/s/d/q/e). Everything else still reaches the browser.
DEFAULT_KEYS = tuple(_SPECIAL) + tuple("abcdefghijklmnopqrstuvwxyz")

# Runs from an <img onerror> handler: viser inserts add_html content with
# innerHTML, which executes inline event handlers but not <script> tags.
_JS = """
if(!window.__so101oinkkeys){window.__so101oinkkeys=true;
var KEYS=new Set(__KEYS__),ws=null,st=document.getElementById('so101oinkkeys');
function show(s,c){if(st){st.textContent=s;st.style.color=c;}}
function connect(){ws=new WebSocket('ws://'+location.hostname+':__PORT__');
  ws.onopen=function(){show('keyboard: connected (click the 3D view, then type)','#2a2');};
  ws.onclose=function(){ws=null;show('keyboard: disconnected, retrying','#c33');setTimeout(connect,1000);};}
connect();
function send(t,k){if(ws&&ws.readyState==1)ws.send(JSON.stringify({t:t,k:k}));}
function form(el){return el&&(el.tagName=='INPUT'||el.tagName=='TEXTAREA'||el.isContentEditable);}
function handler(t){return function(e){
  var k=e.key.toLowerCase();
  if(form(e.target)||!KEYS.has(k))return;
  e.preventDefault();e.stopImmediatePropagation();
  if(t=='d'&&e.repeat)return;
  send(t,k);};}
window.addEventListener('keydown',handler('d'),true);
window.addEventListener('keyup',handler('u'),true);
window.addEventListener('blur',function(){send('blur','');});
}"""


def _to_key(name: str) -> str | None:
    if name in _SPECIAL:
        return _SPECIAL[name]
    return name if len(name) == 1 else None


class BrowserKeys:
    _instances: dict[int, "BrowserKeys"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, port: int) -> "BrowserKeys":
        """The shared server requested on ``port`` (started on first use).

        If ``port`` is busy the server listens on the next free one; check
        ``.port`` for where it ended up.
        """
        with cls._instances_lock:
            if port not in cls._instances:
                inst = cls(port)
                cls._instances[port] = cls._instances[inst.port] = inst  # reachable by either port
            return cls._instances[port]

    def __init__(self, port: int, max_tries: int = 100):
        self._requested_port = port
        self._held: set = set()
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()
        self._attached: set[int] = set()
        for attempt in range(port, port + max_tries):
            try:
                self._server = serve(self._handle, "", attempt)
                break
            except OSError:  # port not available
                continue
        else:
            raise OSError(f"no free port for the browser key server in {port}..{port + max_tries - 1}")
        self.port: int = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True, name="browser-keys").start()

    # ----------------------------------------------------------- public API

    def listener(self, on_press: Callable | None = None, on_release: Callable | None = None) -> Subscription:
        return Subscription(self, on_press, on_release)

    def attach(self, server, keys: tuple[str, ...] = DEFAULT_KEYS) -> None:
        """Inject the key-capture snippet into a ``viser.ViserServer`` GUI."""
        if id(server) in self._attached:
            return
        self._attached.add(id(server))
        js = _JS.replace("__PORT__", str(self.port)).replace("__KEYS__", json.dumps(list(keys)))
        js = js.replace('"', "&quot;")
        server.gui.add_html(
            f'<div style="font-size:0.8em"><span id="so101oinkkeys">keyboard: connecting</span>'
            f'<img src="data:," style="display:none" onerror="this.remove();{js}"></div>'
        )

    def stop(self) -> None:
        self._server.shutdown()
        with self._instances_lock:
            self._instances.pop(self._requested_port, None)
            self._instances.pop(self.port, None)

    # ------------------------------------------------------------ internals

    def _subscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.append(sub)

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def _fire(self, event: str, key) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            cb = sub.on_press if event == "d" else sub.on_release
            if cb is not None:
                cb(key)

    def _handle(self, conn) -> None:
        try:
            for raw in conn:
                msg = json.loads(raw)
                if msg["t"] == "blur":  # the tab lost focus: keyups were swallowed
                    released, self._held = self._held, set()
                    for key in released:
                        self._fire("u", key)
                    continue
                key = _to_key(msg["k"])
                if key is None:
                    continue
                if msg["t"] == "d" and key not in self._held:
                    self._held.add(key)
                    self._fire("d", key)
                elif msg["t"] == "u" and key in self._held:
                    self._held.discard(key)
                    self._fire("u", key)
        finally:
            # a closed tab releases everything it was holding
            released, self._held = self._held, set()
            for key in released:
                self._fire("u", key)

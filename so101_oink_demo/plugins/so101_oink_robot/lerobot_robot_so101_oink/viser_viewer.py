"""Browser-based MuJoCo viewer (mjviser) with the subset of the native
``mujoco.viewer`` passive-viewer API that the SO-101 demo uses.

Nothing here needs a display or OpenGL: the scene is served over HTTP and
rendered in the browser, so it works on a remote machine: open
http://<host>:<port> from your own. The port is the requested one or, if that
is taken (another user's viewer), the next free one; it is printed at startup.
"""

import socket

import mujoco
import viser
from mjviser import ViserMujocoScene


class ViserViewer:
    """Passive viewer: the caller steps the sim and calls ``sync()``."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, port: int = 8080):
        self._data = data
        self.server = viser.ViserServer(port=port, verbose=False)
        self.scene = ViserMujocoScene(self.server, model, num_envs=1)
        # mjviser's default keeps the first jointed body centered, which here is
        # an arm link, so the whole scene would slide around as the arm moves.
        self.scene.camera_tracking_enabled = False
        # Same vantage point as the native viewer (<visual><global> in scene.xml).
        self.scene.create_visualization_gui(
            camera_distance=1.2 * model.stat.extent,
            camera_azimuth=float(model.vis.global_.azimuth),
            camera_elevation=-float(model.vis.global_.elevation),
        )
        self.sync()
        port = self.server.get_port()
        print(
            f"viser viewer: open http://localhost:{port}  (from another machine: http://{socket.gethostname()}:{port})"
        )

    def sync(self) -> None:
        self.scene.update_from_mjdata(self._data)

    def is_running(self) -> bool:
        return True  # no window to close; the process decides when to stop

    def close(self) -> None:
        self.server.stop()

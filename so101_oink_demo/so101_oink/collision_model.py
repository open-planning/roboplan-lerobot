"""Box collision geometry for the self-collision barrier.

The barrier needs link-to-link *distances* at every solver step, and the mesh
collision geometry of the roboplan SO-101 URDF makes that query cost ~100 ms.
Boxes cost microseconds, so the URDF is rewritten on load: each arm link's
collision meshes become their bounding boxes (in the link frame), and the
gripper and moving jaw take the hand-fitted boxes of the MuJoCo model, whose
bodies share the URDF link frames. The boxes are conservative, so the barrier
only ever errs on the safe side.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

MJCF_PATH = Path(__file__).parent / "model" / "so101_new_calib.xml"
MJCF_BOX_LINKS = {"gripper": "gripper_link", "moving_jaw_so101_v1": "moving_jaw_so101_v1_link"}

Box = tuple[np.ndarray, np.ndarray]  # center, full size (link frame)


def _mesh_boxes(urdf: ET.Element, package_dir: Path) -> dict[str, list[Box]]:
    """One bounding box per URDF collision mesh."""
    boxes: dict[str, list[Box]] = {}
    for link in urdf.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            origin = collision.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            path = package_dir / mesh.get("filename").removeprefix("package://")
            points = trimesh.load(path, force="mesh").vertices @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            lo, hi = points.min(axis=0), points.max(axis=0)
            boxes.setdefault(link.get("name"), []).append(((lo + hi) / 2, hi - lo))
    return boxes


def _mjcf_boxes() -> dict[str, list[Box]]:
    """Box geoms of the gripper and jaw bodies (MuJoCo sizes are half-sizes)."""
    boxes: dict[str, list[Box]] = {}
    for body in ET.parse(MJCF_PATH).getroot().iter("body"):
        link = MJCF_BOX_LINKS.get(body.get("name"))
        for geom in body.findall("geom") if link else []:
            if geom.get("type") == "box":
                half = np.array([float(v) for v in geom.get("size").split()])
                pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
                boxes.setdefault(link, []).append((pos, 2 * half))
    return boxes


@lru_cache(maxsize=None)
def box_collision_urdf(urdf_path: Path, package_dir: Path) -> str:
    """The URDF's XML with every link's mesh collisions replaced by boxes."""
    root = ET.parse(urdf_path).getroot()
    boxes = {**_mesh_boxes(root, package_dir), **_mjcf_boxes()}
    for link in root.findall("link"):
        if link.get("name") not in boxes:
            continue
        for collision in link.findall("collision"):
            link.remove(collision)
        for center, size in boxes[link.get("name")]:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", xyz=" ".join(f"{v:.6f}" for v in center), rpy="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=" ".join(f"{v:.6f}" for v in size))
    return ET.tostring(root, encoding="unicode")

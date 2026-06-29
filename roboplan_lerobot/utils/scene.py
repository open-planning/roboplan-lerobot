"""MuJoCo scene assembly and geometry helpers."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np


def build_scene_xml(mjcf_path, scene_mjcf_path) -> str:
    """Load the robot MJCF, fix asset paths, and merge in the scene objects XML.

    from_xml_string has no file-system context, so relative meshdir/texturedir
    paths in gen3.xml would fail. We parse both files, rewrite those paths to
    absolute, then merge scene_mjcf_path's worldbody children before serialising.
    Pass scene_mjcf_path=None to get the bare robot with no extra geometry.
    """
    gen3_dir = mjcf_path.resolve().parent

    root = ET.parse(str(mjcf_path)).getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            val = compiler.get(attr, "")
            if val and not os.path.isabs(val):
                compiler.set(attr, str(gen3_dir / val))

    if scene_mjcf_path is not None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        scene_worldbody = ET.parse(str(scene_mjcf_path)).getroot().find("worldbody")
        if scene_worldbody is not None:
            for child in scene_worldbody:
                worldbody.append(child)

    return ET.tostring(root, encoding="unicode")


def cylinder_mesh(radius: float, height: float, n: int = 20):
    """Closed cylinder mesh centred at the origin along z. Returns (vertices, faces)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    verts = np.vstack([
        np.column_stack([radius * c, radius * s, np.full(n, -height / 2)]),
        np.column_stack([radius * c, radius * s, np.full(n,  height / 2)]),
        [[0.0, 0.0, -height / 2]],
        [[0.0, 0.0,  height / 2]],
    ]).astype(np.float32)
    faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    bc, tc = 2 * n, 2 * n + 1
    for i in range(n):
        faces += [[bc, (i + 1) % n, i], [tc, n + i, n + (i + 1) % n]]
    return verts, np.array(faces, dtype=np.uint32)

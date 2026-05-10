"""
Warp OpenGL headless renderer for MJPEG stream.

Renders 240k particles + ice sheet (CAD) + falling cube.
"""
import io
import os
import math
import numpy as np

os.environ["PYGLET_HEADLESS"] = "1"
os.environ.pop("DISPLAY", None)
import pyglet  # noqa: E402
pyglet.options["headless"] = True

import warp as wp  # noqa: E402
from warp.render import OpenGLRenderer  # noqa: E402
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE  # noqa: E402
from PIL import Image  # noqa: E402

from .config import SCALE, ICE_L, ICE_W, ICE_H, ICE_SHEET, PARTICLE_R  # noqa: E402

WIDTH = 1280
HEIGHT = 720


class WarpParticleRenderer:
    def __init__(self):
        self.frame_count = 0
        self._ice_sheet_verts, self._ice_sheet_faces = self._build_ice_sheet_mesh()

        self.cam_azimuth = 0.076
        self.cam_elevation = 1.305
        self.cam_distance = 21.0
        self.cam_target = [0.0, 0.0, ICE_H * 0.5]

        self._renderer = OpenGLRenderer(
            screen_width=WIDTH,
            screen_height=HEIGHT,
            headless=True,
            draw_grid=False,
            draw_sky=True,
            draw_axis=False,
            show_info=False,
            near_plane=0.01,
            far_plane=100.0,
            camera_fov=40.0,
            camera_pos=(0.0, ICE_H + 6.0, 21.0),
            camera_front=(0.0, -0.3, -1.0),
            camera_up=(0.0, 1.0, 0.0),
            up_axis="Z",
            vsync=False,
            enable_mouse_interaction=False,
            enable_keyboard_interaction=False,
        )
        print("[renderer] Ready")

    def _build_ice_sheet_mesh(self):
        """Load ice sheet from CAD STL, fall back to procedural."""
        stl_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ice-pool-cad.stl"
        )
        try:
            from stl import mesh as stl_mesh
            m = stl_mesh.Mesh.from_file(stl_path)
            raw = m.vectors.reshape(-1, 3).copy()

            min_xyz = raw.min(axis=0)
            max_xyz = raw.max(axis=0)
            center_xy = (min_xyz[:2] + max_xyz[:2]) / 2.0
            raw[:, 0] -= center_xy[0]
            raw[:, 1] -= center_xy[1]
            raw[:, 2] -= max_xyz[2]

            # Swap x<->y (STL convention -> sim convention)
            tmp = raw[:, 0].copy()
            raw[:, 0] = raw[:, 1]
            raw[:, 1] = tmp

            scale_factor = SCALE / 1000.0
            raw *= scale_factor
            raw[:, 2] += ICE_H

            # Double-sided triangles
            raw_tris = raw.reshape(-1, 3, 3)
            raw = np.concatenate([raw_tris, raw_tris[:, ::-1, :]], axis=0).reshape(-1, 3)

            sheet_verts = raw.astype(np.float32)
            sheet_faces = np.arange(len(sheet_verts), dtype=np.int32)
            print(f"[renderer] CAD ice sheet: {len(m.vectors)} tris")
            return sheet_verts, sheet_faces

        except Exception as e:
            print(f"[renderer] STL load failed ({e}), using procedural mesh")
            return self._build_procedural_ice_sheet()

    def _build_procedural_ice_sheet(self):
        """Fallback: procedural ice sheet with rectangular pool."""
        outer_l = 0.500 * SCALE
        outer_w = 0.500 * SCALE
        pool_l = ICE_L
        pool_w = ICE_W
        top_z = ICE_H
        bottom_z = ICE_H - ICE_SHEET

        side_w = (outer_w - pool_w) / 2.0
        end_l = (outer_l - pool_l) / 2.0
        boxes = [
            (outer_l, side_w, top_z, bottom_z, (0.0, -(pool_w / 2.0 + side_w / 2.0))),
            (outer_l, side_w, top_z, bottom_z, (0.0, pool_w / 2.0 + side_w / 2.0)),
            (end_l, pool_w, top_z, bottom_z, (-(pool_l / 2.0 + end_l / 2.0), 0.0)),
            (end_l, pool_w, top_z, bottom_z, (pool_l / 2.0 + end_l / 2.0, 0.0)),
        ]

        verts = []
        faces = []
        for length, width, z1, z0, center in boxes:
            cx, cy = center
            x0 = cx - length / 2.0
            x1 = cx + length / 2.0
            y0 = cy - width / 2.0
            y1 = cy + width / 2.0
            base = len(verts)
            verts.extend([
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ])
            faces.extend([
                base + 0, base + 1, base + 2, base + 0, base + 2, base + 3,
                base + 4, base + 6, base + 5, base + 4, base + 7, base + 6,
                base + 0, base + 4, base + 5, base + 0, base + 5, base + 1,
                base + 1, base + 5, base + 6, base + 1, base + 6, base + 2,
                base + 2, base + 6, base + 7, base + 2, base + 7, base + 3,
                base + 3, base + 7, base + 4, base + 3, base + 4, base + 0,
            ])

        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

    def _build_cube_mesh(self, center, size):
        cx, cy, cz = center
        h = size * 0.5
        verts = np.array([
            (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
            (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
        ], dtype=np.float32)

        angle = math.pi / 4.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated = verts.copy()
        y_new = rotated[:, 1] * cos_a - rotated[:, 2] * sin_a
        z_new = rotated[:, 1] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 1] = y_new
        rotated[:, 2] = z_new
        x_new = rotated[:, 0] * cos_a + rotated[:, 2] * sin_a
        z_new2 = -rotated[:, 0] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 0] = x_new
        rotated[:, 2] = z_new2
        verts = rotated
        verts[:, 0] += cx
        verts[:, 1] += cy
        verts[:, 2] += cz
        faces = np.array([
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ], dtype=np.int32)
        return verts, faces

    def _compute_colors(self, ice_np, pen_np):
        """Particle colors: blue gradient by depth, red ramp for contacts."""
        n = len(ice_np)
        colors = np.empty((n, 3), dtype=np.float32)

        contact_mask = pen_np > 0
        no_contact = ~contact_mask

        if np.any(contact_mask):
            max_pen = max(float(pen_np.max()), 0.001)
            t = np.clip(pen_np[contact_mask] / max_pen, 0.0, 1.0)
            colors[contact_mask, 0] = 0.8 + 0.2 * t
            colors[contact_mask, 1] = 0.3 - 0.2 * t
            colors[contact_mask, 2] = 0.2 - 0.1 * t

        if np.any(no_contact):
            depth_frac = np.clip(ice_np[no_contact, 2] / ICE_H, 0.0, 1.0)
            colors[no_contact, 0] = 0.2 + 0.3 * depth_frac
            colors[no_contact, 1] = 0.5 + 0.3 * depth_frac
            colors[no_contact, 2] = 0.8 + 0.2 * depth_frac

        return colors

    def render_frame(self, physics) -> bytes:
        self.frame_count += 1

        ice_np = physics.ice_pos.numpy()
        pen_np = physics.pen_out.numpy()
        colors = self._compute_colors(ice_np, pen_np)

        az = self.cam_azimuth
        el = self.cam_elevation
        dist = self.cam_distance
        tx, ty, tz = self.cam_target

        cam_x = tx + dist * math.cos(el) * math.sin(az)
        cam_y = ty + dist * math.cos(el) * math.cos(az)
        cam_z = tz + dist * math.sin(el)

        dx = tx - cam_x
        dy = ty - cam_y
        dz = tz - cam_z
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d > 0:
            dx /= d; dy /= d; dz /= d

        self._renderer.update_view_matrix(
            cam_pos=(cam_x, cam_y, cam_z),
            cam_front=(dx, dy, dz),
        )

        self._renderer.begin_frame(self.frame_count * 0.033)

        self._renderer.render_mesh(
            "ice_sheet",
            self._ice_sheet_verts,
            self._ice_sheet_faces,
            colors=(0.72, 0.9, 1.0),
        )

        self._renderer.render_points(
            "ice_particles",
            ice_np,
            radius=PARTICLE_R,
            colors=colors,
        )

        cube_verts, cube_faces = self._build_cube_mesh(physics.cube_pos.numpy()[0], physics.cube_size)
        self._renderer.render_mesh(
            "falling_cube",
            cube_verts,
            cube_faces,
            colors=(1.0, 0.55, 0.12),
        )

        self._renderer.end_frame()

        pixels = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        img_data = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]
        buf = io.BytesIO()
        Image.fromarray(img_data).save(buf, format='JPEG', quality=80)
        return buf.getvalue()

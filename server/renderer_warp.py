"""
Warp OpenGL headless renderer for MJPEG stream.

Uses Warp's built-in OpenGLRenderer with EGL headless mode for
hardware-accelerated 3D rendering of 240k particles + blade mesh.
Requires: pyglet>=2.1, PyOpenGL, NVIDIA EGL drivers.
"""
import io
import os
import math
import numpy as np

# Force EGL headless before importing pyglet
os.environ["PYGLET_HEADLESS"] = "1"
os.environ.pop("DISPLAY", None)
import pyglet  # noqa: E402
pyglet.options["headless"] = True

import warp as wp  # noqa: E402
from warp.render import OpenGLRenderer  # noqa: E402
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE  # noqa: E402
from PIL import Image  # noqa: E402

from .config import SCALE, ICE_L, ICE_W, ICE_H, ICE_SHEET, BLADE_LEN, BLADE_W, BLADE_H  # noqa: E402

WIDTH = 1280
HEIGHT = 720

# Pre-compute color lookup table (256 entries, float RGB)
_lut = np.zeros((256, 3), dtype=np.float32)
for _i in range(256):
    _t = _i / 255.0
    if _t < 0.5:
        _s = _t * 2.0
        _lut[_i] = (0.12 + 0.71 * _s, 0.47 - 0.24 * _s, 0.86 - 0.71 * _s)
    else:
        _s = (_t - 0.5) * 2.0
        _lut[_i] = (0.82 + 0.18 * _s, 0.24 - 0.20 * _s, 0.16 - 0.12 * _s)
COLOR_LUT_NP = _lut


class WarpParticleRenderer:
    """3D GPU-accelerated particle renderer using Warp's OpenGLRenderer."""

    def __init__(self):
        self.frame_count = 0
        self._mesh_loaded = False
        self._mesh_verts = None
        self._mesh_faces = None

        # Orbit camera state (controllable from browser)
        self.cam_azimuth = 1.5708   # radians, horizontal angle (π/2 = side view)
        self.cam_elevation = 0.6    # radians, vertical angle (0=level, pi/2=top)
        self.cam_distance = BLADE_LEN * 1.5  # distance from target
        self.cam_target = [0.0, 0.0, ICE_H * 0.5]  # look-at point

        cam_dist = BLADE_LEN * 1.5
        self._renderer = OpenGLRenderer(
            screen_width=WIDTH,
            screen_height=HEIGHT,
            headless=True,
            draw_grid=True,
            draw_sky=True,
            draw_axis=False,
            show_info=False,
            near_plane=0.01,
            far_plane=100.0,
            camera_fov=40.0,
            camera_pos=(0.0, ICE_H + BLADE_H * 4.0, cam_dist),
            camera_front=(0.0, -0.3, -1.0),
            camera_up=(0.0, 1.0, 0.0),
            up_axis="Z",
            vsync=False,
            enable_mouse_interaction=False,
            enable_keyboard_interaction=False,
        )
        print("[warp-renderer] OpenGL headless renderer initialized")

    def _load_mesh(self, physics):
        if self._mesh_loaded or physics.mesh_data is None:
            return
        self._mesh_verts = np.array(physics.mesh_data['vertices'], dtype=np.float32)
        self._mesh_faces = np.array(physics.mesh_data['faces'], dtype=np.int32)
        self._mesh_loaded = True
        print(f"[warp-renderer] Mesh loaded: {len(self._mesh_verts)} verts, {len(self._mesh_faces)} tris")

    def _compute_colors(self, ice_np, pen_np):
        """Vectorized particle coloring: penetration → red ramp, rest → blue by depth."""
        n = len(ice_np)
        colors = np.empty((n, 3), dtype=np.float32)

        contact_mask = pen_np > 0
        max_pen = max(float(pen_np.max()), 0.001)

        # Contact particles: use penetration color LUT
        if np.any(contact_mask):
            ci = np.clip((pen_np[contact_mask] / max_pen * 255).astype(np.int32), 0, 255)
            colors[contact_mask] = COLOR_LUT_NP[ci]

        # Non-contact particles: blue gradient by depth
        no_contact = ~contact_mask
        if np.any(no_contact):
            depth_frac = np.clip(ice_np[no_contact, 2] / ICE_H, 0.0, 1.0)
            colors[no_contact, 0] = 0.2 + 0.3 * depth_frac
            colors[no_contact, 1] = 0.5 + 0.3 * depth_frac
            colors[no_contact, 2] = 0.8 + 0.2 * depth_frac

        return colors

    def render_frame(self, physics) -> bytes:
        self.frame_count += 1
        self._load_mesh(physics)

        ice_np = physics.ice_pos.numpy()
        pen_np = physics.pen_out.numpy()

        bx, by, bz = physics.pos[0], physics.pos[1], physics.pos[2]
        blade_dir = physics.yaw + physics.alpha
        lean = physics.lean

        colors = self._compute_colors(ice_np, pen_np)

        # Orbit camera: compute position from spherical coordinates
        az = self.cam_azimuth
        el = self.cam_elevation
        dist = self.cam_distance
        tx, ty, tz = self.cam_target

        # Follow blade center for target
        tx, ty = bx, by
        tz = ICE_H * 0.5

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

        sim_time = self.frame_count * 0.033
        self._renderer.begin_frame(sim_time)

        # Particles
        particle_radius = 0.0005 * SCALE  # ~0.025mm real (tiny dots)
        self._renderer.render_points(
            "ice_particles",
            ice_np,
            radius=particle_radius,
            colors=colors,
        )

        # Blade mesh
        if self._mesh_loaded:
            cos_dir = math.cos(blade_dir)
            sin_dir = math.sin(blade_dir)
            cos_lean = math.cos(lean)
            sin_lean = math.sin(lean)

            verts = self._mesh_verts.copy()
            # Apply lean rotation (around blade's X axis in local frame)
            y_rot = verts[:, 1] * cos_lean - verts[:, 2] * sin_lean
            z_rot = verts[:, 1] * sin_lean + verts[:, 2] * cos_lean
            verts[:, 1] = y_rot
            verts[:, 2] = z_rot

            # Rotate to world frame (around Z axis)
            x_w = bx + verts[:, 0] * cos_dir - verts[:, 1] * sin_dir
            y_w = by + verts[:, 0] * sin_dir + verts[:, 1] * cos_dir
            z_w = bz + verts[:, 2]

            world_verts = np.column_stack([x_w, y_w, z_w]).astype(np.float32)
            self._renderer.render_mesh(
                "blade",
                world_verts,
                self._mesh_faces.flatten(),
                colors=(0.75, 0.75, 0.85),
            )

        self._renderer.end_frame()

        # Read pixels
        pixels = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        img_data = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]

        pil_img = Image.fromarray(img_data)
        # Overlay camera values as text for debugging
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(pil_img)
            cam_text = f"az={self.cam_azimuth:.3f} el={self.cam_elevation:.3f} d={self.cam_distance:.1f} f={self.frame_count}"
            draw.rectangle([(5, 5), (450, 25)], fill=(0, 0, 0))
            draw.text((8, 7), cam_text, fill=(255, 255, 0))
        except Exception:
            pass
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()

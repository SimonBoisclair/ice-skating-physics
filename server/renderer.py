"""
Headless particle renderer for MJPEG stream.

Renders 240k ice particles + actual blade CAD mesh to a JPEG frame using numpy + PIL.
Three views: top-down (XY), side (XZ along blade), front (YZ cross-section).
Particles colored by penetration depth (blue=surface, red=deep).
"""
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SCALE, ICE_L, ICE_W, ICE_H, BLADE_LEN, BLADE_W, BLADE_H


# ── rendering config ──────────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 720
BG_COLOR = (15, 15, 25)
ICE_SURFACE_COLOR = (40, 60, 80)
BLADE_COLOR = (220, 220, 240)
BLADE_FILL = (40, 50, 70)
TEXT_COLOR = (200, 210, 220)
LABEL_COLOR = (140, 150, 170)

# Particle color ramp: surface (blue) -> deep (red)
def _pen_color(depth_frac):
    t = max(0.0, min(1.0, depth_frac))
    if t < 0.5:
        s = t * 2.0
        r = int(30 + 180 * s)
        g = int(120 - 60 * s)
        b = int(220 - 180 * s)
    else:
        s = (t - 0.5) * 2.0
        r = int(210 + 45 * s)
        g = int(60 - 50 * s)
        b = int(40 - 30 * s)
    return (r, g, b)

COLOR_LUT = [_pen_color(i / 255.0) for i in range(256)]


def _convex_hull_2d(points):
    """Simple 2D convex hull (Andrew's monotone chain). Returns ordered hull points."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


class ParticleRenderer:
    """Renders particle state to JPEG frames for MJPEG streaming."""

    def __init__(self):
        self.frame_count = 0
        self._mesh_tris_xz = None  # cached projected triangles for side view
        self._mesh_tris_xy = None  # cached projected triangles for top view
        self._mesh_tris_yz = None  # cached projected triangles for front view
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            self.font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
        except Exception:
            self.font = ImageFont.load_default()
            self.font_small = self.font
            self.font_title = self.font

    def _get_mesh_tris(self, physics):
        """Extract projected 2D triangles from blade mesh for each view (cached)."""
        if self._mesh_tris_xz is not None:
            return

        if physics.mesh_data is None:
            return

        verts = np.array(physics.mesh_data['vertices'])  # blade-local: X=along, Y=thickness, Z=height
        faces = np.array(physics.mesh_data['faces'])     # triangle indices

        # Pre-compute projected triangles for each view
        # Each entry is a list of [(x0,y0), (x1,y1), (x2,y2)] triangles
        self._mesh_tris_xz = []  # Side view: (X, Z) = (along, height)
        self._mesh_tris_xy = []  # Top view: (X, Y) = (along, thickness)
        self._mesh_tris_yz = []  # Front view: (Y, Z) = (thickness, height)

        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            self._mesh_tris_xz.append(((v0[0], v0[2]), (v1[0], v1[2]), (v2[0], v2[2])))
            self._mesh_tris_xy.append(((v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])))
            self._mesh_tris_yz.append(((v0[1], v0[2]), (v1[1], v1[2]), (v2[1], v2[2])))

    def render_frame(self, physics) -> bytes:
        """Render current physics state to JPEG bytes."""
        self.frame_count += 1
        self._get_mesh_tris(physics)

        # Read particle data from GPU
        ice_np = physics.ice_pos.numpy()
        pen_np = physics.pen_out.numpy()

        # Blade state
        bx, by, bz = physics.pos[0], physics.pos[1], physics.pos[2]
        blade_dir = physics.yaw + physics.alpha
        lean = physics.lean

        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Layout: 3 panels in a row
        # Left: top-down (XY)  |  Middle: side (XZ)  |  Right: front (YZ)
        hud_h = 50
        panel_h = HEIGHT - hud_h - 10
        panel_w = (WIDTH - 20) // 3  # 3 equal panels

        x0 = 5
        x1 = x0 + panel_w + 5
        x2 = x1 + panel_w + 5

        self._draw_top_view(draw, ice_np, pen_np, bx, by, bz, blade_dir, lean,
                            x0, 5, panel_w, panel_h)
        self._draw_side_view(draw, ice_np, pen_np, bx, by, bz, blade_dir, lean,
                             x1, 5, panel_w, panel_h)
        self._draw_front_view(draw, ice_np, pen_np, bx, by, bz, blade_dir, lean,
                              x2, 5, panel_w, panel_h)

        # HUD
        self._draw_hud(draw, physics, HEIGHT - hud_h)

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=78)
        return buf.getvalue()

    def _draw_mesh_triangles(self, draw, tris, to_screen_fn,
                             ox, oy, pw, ph, fill=BLADE_FILL, outline=BLADE_COLOR):
        """Draw exact mesh silhouette by rendering all projected triangles."""
        if tris is None or len(tris) == 0:
            return
        for tri in tris:
            pts = [to_screen_fn(tri[0][0], tri[0][1]),
                   to_screen_fn(tri[1][0], tri[1][1]),
                   to_screen_fn(tri[2][0], tri[2][1])]
            # Skip degenerate triangles
            if pts[0] == pts[1] or pts[1] == pts[2] or pts[0] == pts[2]:
                continue
            # Quick bounds check - at least one vertex in panel
            if any(ox - 50 <= x <= ox + pw + 50 and oy - 50 <= y <= oy + ph + 50 for x, y in pts):
                draw.polygon(pts, fill=fill, outline=fill)

    def _draw_top_view(self, draw, ice_np, pen_np, bx, by, bz,
                       blade_dir, lean, ox, oy, pw, ph):
        """Top-down view in blade-local frame (along-blade horizontal, across-blade vertical)."""
        draw.rectangle([ox, oy, ox + pw, oy + ph], fill=(20, 22, 30))
        draw.text((ox + 5, oy + 3), "TOP (along / across)", fill=LABEL_COLOR, font=self.font_small)

        margin = 15
        view_l = ICE_L * 1.2   # along blade
        view_w = ICE_W * 5.0   # across blade
        scale_x = (pw - 2 * margin) / view_l
        scale_y = (ph - 2 * margin) / view_w
        scale = min(scale_x, scale_y)

        cx = ox + pw // 2
        cy = oy + margin + (ph - 2 * margin) // 2

        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)

        def to_screen_local(along, across):
            """Map blade-local (along, across) to screen pixels."""
            sx = cx + along * scale
            sy = cy - across * scale
            return int(sx), int(sy)

        def world_to_local(px, py):
            """Project world (px, py) into blade-local frame."""
            dx = px - bx
            dy = py - by
            along = dx * cos_b + dy * sin_b
            across = -dx * sin_b + dy * cos_b
            return along, across

        # Ice field boundary (in blade-local frame)
        # Pool is axis-aligned in world, so compute its corners in local frame
        half_l = ICE_L / 2
        half_w = ICE_W / 2
        corners_world = [
            (bx - half_l, by - half_w), (bx + half_l, by - half_w),
            (bx + half_l, by + half_w), (bx - half_l, by + half_w),
        ]
        corners_local = [world_to_local(cx_w, cy_w) for cx_w, cy_w in corners_world]
        screen_corners = [to_screen_local(a, c) for a, c in corners_local]
        draw.polygon(screen_corners, outline=ICE_SURFACE_COLOR)

        # Particles in blade-local frame
        n = len(ice_np)
        step = max(1, n // 25000)
        max_pen = max(float(pen_np.max()), 0.001)

        for i in range(0, n, step):
            px, py, pz = ice_np[i]
            along, across = world_to_local(px, py)
            sx, sy = to_screen_local(along, across)
            if ox <= sx <= ox + pw and oy <= sy <= oy + ph:
                pen_val = pen_np[i]
                if pen_val > 0:
                    ci = min(255, int(pen_val / max_pen * 255))
                    draw.point((sx, sy), fill=COLOR_LUT[ci])
                else:
                    depth_frac = max(0, min(1, pz / ICE_H))
                    g = int(50 + 50 * depth_frac)
                    draw.point((sx, sy), fill=(g, g, int(g * 1.2)))

        # Draw blade mesh (exact triangles, already in blade-local: X=along, Y=thickness)
        if self._mesh_tris_xy is not None:
            def mesh_to_screen(along, across):
                return to_screen_local(along, across)
            self._draw_mesh_triangles(draw, self._mesh_tris_xy, mesh_to_screen,
                                      ox, oy, pw, ph)
        else:
            # Fallback: horizontal line
            half_l = BLADE_LEN / 2
            x0s, y0s = to_screen_local(-half_l, 0)
            x1s, y1s = to_screen_local(half_l, 0)
            draw.line([(x0s, y0s), (x1s, y1s)], fill=BLADE_COLOR, width=2)

        # Blade center marker
        scx, scy = to_screen_local(0, 0)
        draw.ellipse([scx-3, scy-3, scx+3, scy+3], fill=(255, 100, 100))

    def _draw_side_view(self, draw, ice_np, pen_np, bx, by, bz,
                        blade_dir, lean, ox, oy, pw, ph):
        """Side view along blade (along-axis horizontal, Z vertical)."""
        draw.rectangle([ox, oy, ox + pw, oy + ph], fill=(20, 22, 30))
        draw.text((ox + 5, oy + 3), "SIDE (along-Z)", fill=LABEL_COLOR, font=self.font_small)

        margin = 15
        view_l = BLADE_LEN * 1.3
        view_h = ICE_H * 10.0
        scale_x = (pw - 2 * margin) / view_l
        scale_z = (ph - 2 * margin) / view_h
        scale = min(scale_x, scale_z)

        cx = ox + pw // 2
        surface_y = oy + int((ph - 2 * margin) * 0.3) + margin

        def to_screen(along, z):
            sx = cx + along * scale
            sy = surface_y - (z - ICE_H) * scale
            return int(sx), int(sy)

        # Ice surface line
        x_left = ox + margin
        x_right = ox + pw - margin
        draw.line([(x_left, surface_y), (x_right, surface_y)], fill=(60, 120, 180), width=2)
        draw.text((x_left, surface_y - 14), "ice surface", fill=(60, 120, 180), font=self.font_small)

        # Ice particle pool band (5mm deep, gradient)
        sy_floor = surface_y + int(ICE_H * scale)
        pool_height = sy_floor - surface_y
        for row in range(pool_height):
            depth_frac = row / max(1, pool_height)
            r = int(20 + 15 * (1 - depth_frac))
            g = int(40 + 30 * (1 - depth_frac))
            b = int(70 + 40 * (1 - depth_frac))
            draw.line([(x_left, surface_y + row), (x_right, surface_y + row)], fill=(r, g, b))

        # Pool floor line
        draw.line([(x_left, sy_floor), (x_right, sy_floor)], fill=(40, 80, 120), width=1)
        draw.text((x_right - 70, sy_floor + 2), "5mm deep", fill=(40, 80, 120), font=self.font_small)

        # Particles projected along blade axis
        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)
        n = len(ice_np)
        step = max(1, n // 25000)
        max_pen = max(float(pen_np.max()), 0.001)

        for i in range(0, n, step):
            px, py, pz = ice_np[i]
            dx = px - bx
            dy = py - by
            along = dx * cos_b + dy * sin_b
            perp = abs(-dx * sin_b + dy * cos_b)
            if perp < ICE_W * 0.8:
                sx, sy = to_screen(along, pz)
                if ox <= sx <= ox + pw and oy <= sy <= oy + ph:
                    pen_val = pen_np[i]
                    if pen_val > 0:
                        ci = min(255, int(pen_val / max_pen * 255))
                        draw.point((sx, sy), fill=COLOR_LUT[ci])
                    else:
                        depth_frac = max(0, min(1, pz / ICE_H))
                        r = int(50 * (1 - depth_frac))
                        g_val = int(180 * (1 - depth_frac) + 40)
                        b_val = int(200 - 80 * depth_frac)
                        draw.point((sx, sy), fill=(r, g_val, b_val))

        # Draw blade mesh (exact triangles, side view: X=along, Z=height)
        if self._mesh_tris_xz is not None:
            def mesh_to_screen(along, height):
                return to_screen(along, bz + height)
            self._draw_mesh_triangles(draw, self._mesh_tris_xz, mesh_to_screen,
                                      ox, oy, pw, ph)
        else:
            # Fallback
            half_l = BLADE_LEN / 2
            edge_z = bz - 0.75 * math.cos(lean)
            x0s, y0s = to_screen(-half_l, edge_z)
            x1s, y1s = to_screen(half_l, edge_z)
            draw.line([(x0s, y0s), (x1s, y1s)], fill=BLADE_COLOR, width=2)

        # Penetration annotation
        edge_z = bz - 0.75 * math.cos(lean)
        if edge_z < ICE_H:
            pen_scaled = ICE_H - edge_z
            pen_mm = pen_scaled / SCALE * 1000
            sx_mid, sy_top = to_screen(0, ICE_H)
            _, sy_edge = to_screen(0, edge_z)
            if sy_edge > sy_top + 5:
                draw.line([(sx_mid + 25, sy_top), (sx_mid + 25, sy_edge)],
                          fill=(255, 100, 100), width=2)
                draw.text((sx_mid + 30, (sy_top + sy_edge) // 2 - 6),
                          f"{pen_mm:.3f}mm", fill=(255, 100, 100), font=self.font_small)

    def _draw_front_view(self, draw, ice_np, pen_np, bx, by, bz,
                         blade_dir, lean, ox, oy, pw, ph):
        """Front view (Y horizontal, Z vertical) — blade cross-section."""
        draw.rectangle([ox, oy, ox + pw, oy + ph], fill=(20, 22, 30))
        draw.text((ox + 5, oy + 3), "FRONT (Y-Z)", fill=LABEL_COLOR, font=self.font_small)

        margin = 15
        legend_w = 30  # width reserved for depth legend on right side
        view_pw = pw - legend_w  # panel width minus legend

        # Zoom to blade thickness + some surroundings
        view_w = BLADE_W * 40.0  # show wider area
        view_h = ICE_H * 10.0
        scale_y = (view_pw - 2 * margin) / view_w
        scale_z = (ph - 2 * margin) / view_h
        scale = min(scale_y, scale_z)

        cx = ox + view_pw // 2
        surface_y = oy + int((ph - 2 * margin) * 0.3) + margin

        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)

        def to_screen(across, z):
            sx = cx + across * scale
            sy = surface_y - (z - ICE_H) * scale
            return int(sx), int(sy)

        # Ice surface line
        x_left = ox + margin
        x_right = ox + view_pw - margin
        draw.line([(x_left, surface_y), (x_right, surface_y)], fill=(60, 120, 180), width=2)
        draw.text((x_left, surface_y - 14), "ice surface (z=0)", fill=(60, 120, 180), font=self.font_small)

        # Ice particle pool band (5mm deep, colored gradient background)
        sy_floor = surface_y + int(ICE_H * scale)
        pool_height = sy_floor - surface_y
        for row in range(pool_height):
            depth_frac = row / max(1, pool_height)
            # Gradient from light blue (top/surface) to dark blue (bottom/deep)
            r = int(20 + 15 * (1 - depth_frac))
            g = int(40 + 30 * (1 - depth_frac))
            b = int(70 + 40 * (1 - depth_frac))
            draw.line([(x_left, surface_y + row), (x_right, surface_y + row)], fill=(r, g, b))

        # Pool floor line
        draw.line([(x_left, sy_floor), (x_right, sy_floor)], fill=(40, 80, 120), width=1)
        draw.text((x_left, sy_floor + 2), "pool floor (5mm)", fill=(40, 80, 120), font=self.font_small)

        # Depth legend on the right side of panel
        leg_x = ox + view_pw + 2
        leg_top = surface_y
        leg_bot = sy_floor
        leg_height = leg_bot - leg_top
        if leg_height > 10:
            draw.text((leg_x, leg_top - 14), "0mm", fill=LABEL_COLOR, font=self.font_small)
            draw.text((leg_x, leg_bot + 2), "5mm", fill=LABEL_COLOR, font=self.font_small)
            # Draw color bar
            for row in range(leg_height):
                depth_frac = row / leg_height
                ci = min(255, int(depth_frac * 255))
                color = COLOR_LUT[ci]
                draw.line([(leg_x, leg_top + row), (leg_x + 12, leg_top + row)], fill=color)

        # Particles projected perpendicular to blade
        n = len(ice_np)
        step = max(1, n // 25000)
        max_pen = max(float(pen_np.max()), 0.001)

        for i in range(0, n, step):
            px, py, pz = ice_np[i]
            dx = px - bx
            dy = py - by
            along = dx * cos_b + dy * sin_b
            across = -dx * sin_b + dy * cos_b
            # Only show particles near blade center (within blade length)
            if abs(along) < BLADE_LEN * 0.3:
                sx, sy = to_screen(across, pz)
                if ox <= sx <= ox + view_pw and oy <= sy <= oy + ph:
                    pen_val = pen_np[i]
                    if pen_val > 0:
                        ci = min(255, int(pen_val / max_pen * 255))
                        draw.point((sx, sy), fill=COLOR_LUT[ci])
                    else:
                        depth_frac = max(0, min(1, pz / ICE_H))
                        # Distinct depth colors: cyan (surface) -> blue (mid) -> dark blue (deep)
                        r = int(50 * (1 - depth_frac))
                        g_val = int(180 * (1 - depth_frac) + 40)
                        b_val = int(200 - 80 * depth_frac)
                        draw.point((sx, sy), fill=(r, g_val, b_val))

        # Draw blade mesh (exact triangles, front view: Y=thickness, Z=height)
        if self._mesh_tris_yz is not None:
            cos_lean = math.cos(lean)
            sin_lean = math.sin(lean)

            def mesh_to_screen(thickness, height):
                # Apply lean rotation: rotate (thickness, height) by lean angle
                rotated_y = thickness * cos_lean - height * sin_lean
                rotated_z = thickness * sin_lean + height * cos_lean
                return to_screen(rotated_y, bz + rotated_z)
            self._draw_mesh_triangles(draw, self._mesh_tris_yz, mesh_to_screen,
                                      ox, oy, pw, ph)
        else:
            # Fallback: vertical line
            edge_z = bz - 0.75 * math.cos(lean)
            top_z = bz + 0.75 * math.cos(lean)
            sx, sy_bot = to_screen(0, edge_z)
            _, sy_top = to_screen(0, top_z)
            draw.line([(sx, sy_bot), (sx, sy_top)], fill=BLADE_COLOR, width=2)

        # Lean angle indicator
        draw.text((ox + pw - 60, oy + ph - 18),
                  f"lean={math.degrees(lean):.1f}\u00b0",
                  fill=LABEL_COLOR, font=self.font_small)

    def _draw_hud(self, draw, physics, y_start):
        """Draw HUD info bar at bottom of frame."""
        y = y_start
        items = [
            f"Frame: {physics.frame}",
            f"Mass: {physics.blade_mass:.0f}kg",
            f"Lean: {math.degrees(physics.lean):.1f}\u00b0",
            f"Pen: {physics.pen_max_mm:.3f}mm",
            f"Contact: {physics.pen_contact_count}",
            f"F_z: {physics.reaction_fz_real:.0f}N",
            f"Paused: {physics.physics_paused}",
        ]
        x = 10
        for item in items:
            draw.text((x, y), item, fill=TEXT_COLOR, font=self.font_small)
            x += len(item) * 7 + 12

        # Color legend
        legend_y = y + 18
        draw.text((10, legend_y), "Penetration:", fill=LABEL_COLOR, font=self.font_small)
        for i in range(100):
            ci = int(i * 2.55)
            color = COLOR_LUT[ci]
            draw.rectangle([110 + i * 2, legend_y, 111 + i * 2, legend_y + 10], fill=color)
        draw.text((110, legend_y + 12), "none", fill=LABEL_COLOR, font=self.font_small)
        draw.text((280, legend_y + 12), "deep", fill=LABEL_COLOR, font=self.font_small)

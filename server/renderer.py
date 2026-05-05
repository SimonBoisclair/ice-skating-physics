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
        self._mesh_hull_xz = None  # cached mesh silhouette for side view
        self._mesh_hull_xy = None  # cached mesh silhouette for top view
        self._mesh_hull_yz = None  # cached mesh silhouette for front view
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            self.font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
        except Exception:
            self.font = ImageFont.load_default()
            self.font_small = self.font
            self.font_title = self.font

    def _get_mesh_hulls(self, physics):
        """Extract 2D silhouette hulls from blade mesh (cached)."""
        if self._mesh_hull_xz is not None:
            return

        if physics.mesh_data is None:
            return

        verts = np.array(physics.mesh_data['vertices'])  # blade-local: X=along, Y=thickness, Z=height
        # Side view: project to (X, Z)
        self._mesh_hull_xz = _convex_hull_2d(verts[:, [0, 2]])
        # Top view: project to (X, Y)
        self._mesh_hull_xy = _convex_hull_2d(verts[:, [0, 1]])
        # Front view: project to (Y, Z)
        self._mesh_hull_yz = _convex_hull_2d(verts[:, [1, 2]])

    def render_frame(self, physics) -> bytes:
        """Render current physics state to JPEG bytes."""
        self.frame_count += 1
        self._get_mesh_hulls(physics)

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

    def _draw_mesh_silhouette(self, draw, hull_points, to_screen_fn,
                              ox, oy, pw, ph, fill=BLADE_FILL, outline=BLADE_COLOR):
        """Draw a mesh silhouette polygon from hull points using to_screen transform."""
        if hull_points is None or len(hull_points) < 3:
            return
        screen_pts = []
        for pt in hull_points:
            sx, sy = to_screen_fn(pt[0], pt[1])
            screen_pts.append((sx, sy))
        # Clip check (at least some points in bounds)
        if any(ox <= x <= ox + pw and oy <= y <= oy + ph for x, y in screen_pts):
            draw.polygon(screen_pts, fill=fill, outline=outline)

    def _draw_top_view(self, draw, ice_np, pen_np, bx, by, bz,
                       blade_dir, lean, ox, oy, pw, ph):
        """Top-down view (X horizontal, Y vertical)."""
        draw.rectangle([ox, oy, ox + pw, oy + ph], fill=(20, 22, 30))
        draw.text((ox + 5, oy + 3), "TOP (X-Y)", fill=LABEL_COLOR, font=self.font_small)

        margin = 15
        view_l = ICE_L * 1.2
        view_w = ICE_W * 5.0
        scale_x = (pw - 2 * margin) / view_l
        scale_y = (ph - 2 * margin) / view_w
        scale = min(scale_x, scale_y)

        cx = ox + pw // 2
        cy = oy + margin + (ph - 2 * margin) // 2

        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)

        def to_screen(x, y):
            sx = cx + (x - bx) * scale
            sy = cy - (y - by) * scale
            return int(sx), int(sy)

        # Ice field boundary
        x0, y0 = to_screen(bx - ICE_L/2, by - ICE_W/2)
        x1, y1 = to_screen(bx + ICE_L/2, by + ICE_W/2)
        draw.rectangle([x0, y1, x1, y0], outline=ICE_SURFACE_COLOR, width=1)

        # Particles
        n = len(ice_np)
        step = max(1, n // 25000)
        max_pen = max(float(pen_np.max()), 0.001)

        for i in range(0, n, step):
            px, py, pz = ice_np[i]
            sx, sy = to_screen(px, py)
            if ox <= sx <= ox + pw and oy <= sy <= oy + ph:
                pen_val = pen_np[i]
                if pen_val > 0:
                    ci = min(255, int(pen_val / max_pen * 255))
                    draw.point((sx, sy), fill=COLOR_LUT[ci])
                else:
                    depth_frac = max(0, min(1, pz / ICE_H))
                    g = int(50 + 50 * depth_frac)
                    draw.point((sx, sy), fill=(g, g, int(g * 1.2)))

        # Draw blade mesh silhouette (top view: X along, Y thickness)
        if self._mesh_hull_xy is not None:
            def mesh_to_screen(along, across):
                # Transform from blade-local to world then to screen
                wx = bx + along * cos_b - across * sin_b
                wy = by + along * sin_b + across * cos_b
                return to_screen(wx, wy)
            self._draw_mesh_silhouette(draw, self._mesh_hull_xy, mesh_to_screen,
                                       ox, oy, pw, ph)
        else:
            # Fallback: simple line
            half_l = BLADE_LEN / 2
            x0s, y0s = to_screen(bx - half_l * cos_b, by - half_l * sin_b)
            x1s, y1s = to_screen(bx + half_l * cos_b, by + half_l * sin_b)
            draw.line([(x0s, y0s), (x1s, y1s)], fill=BLADE_COLOR, width=2)

        # Blade center marker
        scx, scy = to_screen(bx, by)
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
        draw.line([(x_left, surface_y), (x_right, surface_y)], fill=(60, 120, 180), width=1)
        draw.text((x_left, surface_y + 2), "ice surface", fill=(60, 120, 180), font=self.font_small)

        # Ice volume
        sy_floor = surface_y + int(ICE_H * scale)
        draw.rectangle([x_left, surface_y, x_right, sy_floor], fill=(25, 35, 50))

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
                        g = int(40 + 60 * depth_frac)
                        draw.point((sx, sy), fill=(g, g, int(g * 1.2)))

        # Draw blade mesh silhouette (side view: X=along, Z=height)
        if self._mesh_hull_xz is not None:
            def mesh_to_screen(along, height):
                return to_screen(along, bz + height)
            self._draw_mesh_silhouette(draw, self._mesh_hull_xz, mesh_to_screen,
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
        # Zoom to blade thickness + some surroundings
        view_w = BLADE_W * 40.0  # show wider area
        view_h = ICE_H * 10.0
        scale_y = (pw - 2 * margin) / view_w
        scale_z = (ph - 2 * margin) / view_h
        scale = min(scale_y, scale_z)

        cx = ox + pw // 2
        surface_y = oy + int((ph - 2 * margin) * 0.3) + margin

        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)

        def to_screen(across, z):
            sx = cx + across * scale
            sy = surface_y - (z - ICE_H) * scale
            return int(sx), int(sy)

        # Ice surface
        x_left = ox + margin
        x_right = ox + pw - margin
        draw.line([(x_left, surface_y), (x_right, surface_y)], fill=(60, 120, 180), width=1)
        draw.text((x_left, surface_y + 2), "ice surface", fill=(60, 120, 180), font=self.font_small)

        # Ice volume
        sy_floor = surface_y + int(ICE_H * scale)
        draw.rectangle([x_left, surface_y, x_right, sy_floor], fill=(25, 35, 50))

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
                if ox <= sx <= ox + pw and oy <= sy <= oy + ph:
                    pen_val = pen_np[i]
                    if pen_val > 0:
                        ci = min(255, int(pen_val / max_pen * 255))
                        draw.point((sx, sy), fill=COLOR_LUT[ci])
                    else:
                        depth_frac = max(0, min(1, pz / ICE_H))
                        g = int(40 + 60 * depth_frac)
                        draw.point((sx, sy), fill=(g, g, int(g * 1.2)))

        # Draw blade mesh silhouette (front view: Y=thickness, Z=height)
        if self._mesh_hull_yz is not None:
            cos_lean = math.cos(lean)
            sin_lean = math.sin(lean)

            def mesh_to_screen(thickness, height):
                # Apply lean rotation: rotate (thickness, height) by lean angle
                rotated_y = thickness * cos_lean - height * sin_lean
                rotated_z = thickness * sin_lean + height * cos_lean
                return to_screen(rotated_y, bz + rotated_z)
            self._draw_mesh_silhouette(draw, self._mesh_hull_yz, mesh_to_screen,
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
            f"Frame: {self.frame_count}",
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

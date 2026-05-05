"""
Headless particle renderer for MJPEG stream.

Renders 240k ice particles + blade outline to a JPEG frame using numpy + PIL.
No GPU rendering needed — reads particle positions from Warp arrays, does
2D projection with numpy, draws with PIL.

Three camera views: top-down (XY), side (XZ), front (YZ).
Particles colored by Z-depth (blue=surface, red=deep penetration).
"""
import io
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SCALE, ICE_L, ICE_W, ICE_H, BLADE_LEN, BLADE_W, BLADE_H


# ── rendering config ──────────────────────────────────────────────
WIDTH  = 960
HEIGHT = 540
BG_COLOR = (15, 15, 25)
ICE_SURFACE_COLOR = (40, 60, 80)
BLADE_COLOR = (220, 220, 240)
GRID_COLOR = (30, 40, 55)
TEXT_COLOR = (200, 210, 220)
LABEL_COLOR = (140, 150, 170)

# Particle color ramp: surface (blue) → deep (red)
def _pen_color(depth_frac):
    """Map normalized depth [0,1] to RGB color."""
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

# Pre-build a 256-entry color lookup table
COLOR_LUT = [_pen_color(i / 255.0) for i in range(256)]


class ParticleRenderer:
    """Renders particle state to JPEG frames for MJPEG streaming."""

    def __init__(self):
        self.frame_count = 0
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            self.font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
        except Exception:
            self.font = ImageFont.load_default()
            self.font_small = self.font
            self.font_title = self.font

    def render_frame(self, physics) -> bytes:
        """Render current physics state to JPEG bytes."""
        self.frame_count += 1

        # Read particle data from GPU
        ice_np = physics.ice_pos.numpy()      # (N, 3)
        pen_np = physics.pen_out.numpy()       # (N,)

        # Blade state
        bx, by, bz = physics.pos[0], physics.pos[1], physics.pos[2]
        blade_dir = physics.yaw + physics.alpha
        lean = physics.lean

        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Layout: two panels side by side
        # Left: top-down view (XY)     Right: side view (XZ, along blade)
        mid_x = WIDTH // 2
        panel_w = mid_x - 10
        panel_h = HEIGHT - 60  # leave room for HUD at bottom

        self._draw_top_view(draw, ice_np, pen_np, bx, by, bz, blade_dir, lean,
                            5, 5, panel_w, panel_h)
        self._draw_side_view(draw, ice_np, pen_np, bx, by, bz, blade_dir, lean,
                             mid_x + 5, 5, panel_w, panel_h)

        # HUD bar at bottom
        self._draw_hud(draw, physics, panel_h + 10)

        # Encode to JPEG
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return buf.getvalue()

    def _draw_top_view(self, draw, ice_np, pen_np, bx, by, bz,
                       blade_dir, lean, ox, oy, pw, ph):
        """Top-down view (X horizontal, Y vertical)."""
        # Panel background
        draw.rectangle([ox, oy, ox + pw, oy + ph], fill=(20, 22, 30))
        draw.text((ox + 5, oy + 3), "TOP VIEW (X-Y)", fill=LABEL_COLOR, font=self.font_small)

        # Scale: map ice field to panel
        margin = 20
        view_l = ICE_L * 1.1  # slight padding
        view_w = ICE_W * 4.0  # wider view for lean
        scale_x = (pw - 2 * margin) / view_l
        scale_y = (ph - 2 * margin) / view_w
        scale = min(scale_x, scale_y)

        cx = ox + pw // 2
        cy = oy + margin + (ph - 2 * margin) // 2

        def to_screen(x, y):
            sx = cx + (x - bx) * scale
            sy = cy - (y - by) * scale
            return int(sx), int(sy)

        # Draw ice field boundary
        x0, y0 = to_screen(bx - ICE_L/2, by - ICE_W/2)
        x1, y1 = to_screen(bx + ICE_L/2, by + ICE_W/2)
        draw.rectangle([x0, y1, x1, y0], outline=ICE_SURFACE_COLOR, width=1)

        # Draw particles (subsample for speed if needed)
        n = len(ice_np)
        step = max(1, n // 30000)  # draw up to 30k particles
        max_pen = max(float(pen_np.max()), 0.001)

        for i in range(0, n, step):
            px, py, pz = ice_np[i]
            sx, sy = to_screen(px, py)
            if ox <= sx <= ox + pw and oy <= sy <= oy + ph:
                pen_val = pen_np[i]
                if pen_val > 0:
                    ci = min(255, int(pen_val / max_pen * 255))
                    color = COLOR_LUT[ci]
                    draw.point((sx, sy), fill=color)
                else:
                    # Depth-based color for non-penetrating particles
                    depth_frac = max(0, min(1, pz / ICE_H))
                    gray = int(40 + 60 * depth_frac)
                    draw.point((sx, sy), fill=(gray, gray, int(gray * 1.3)))

        # Draw blade outline
        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)
        half_l = BLADE_LEN / 2
        # Blade as a line
        x0, y0 = to_screen(bx - half_l * cos_b, by - half_l * sin_b)
        x1, y1 = to_screen(bx + half_l * cos_b, by + half_l * sin_b)
        draw.line([(x0, y0), (x1, y1)], fill=BLADE_COLOR, width=2)

        # Blade center marker
        scx, scy = to_screen(bx, by)
        draw.ellipse([scx-3, scy-3, scx+3, scy+3], fill=(255, 100, 100))

    def _draw_side_view(self, draw, ice_np, pen_np, bx, by, bz,
                        blade_dir, lean, ox, oy, pw, ph):
        """Side view along blade (X horizontal, Z vertical). Shows penetration depth."""
        draw.rectangle([ox, oy, ox + pw, oy + ph], fill=(20, 22, 30))
        draw.text((ox + 5, oy + 3), "SIDE VIEW (X-Z)", fill=LABEL_COLOR, font=self.font_small)

        margin = 20
        # Zoom into contact zone
        view_l = BLADE_LEN * 1.2
        view_h = ICE_H * 8.0  # show above and below ice surface
        scale_x = (pw - 2 * margin) / view_l
        scale_z = (ph - 2 * margin) / view_h
        scale = min(scale_x, scale_z)

        cx = ox + pw // 2
        # Ice surface at 2/3 height from top
        surface_y = oy + int((ph - 2 * margin) * 0.35) + margin

        def to_screen(along, z):
            sx = cx + along * scale
            sy = surface_y - (z - ICE_H) * scale  # ICE_H = surface
            return int(sx), int(sy)

        # Draw ice surface line
        x0 = ox + margin
        x1 = ox + pw - margin
        sy_surface = surface_y
        draw.line([(x0, sy_surface), (x1, sy_surface)], fill=(60, 120, 180), width=1)
        draw.text((x0, sy_surface + 2), "ice surface", fill=(60, 120, 180), font=self.font_small)

        # Draw ice volume (below surface to floor)
        sy_floor = surface_y + int(ICE_H * scale)
        draw.rectangle([x0, sy_surface, x1, sy_floor], fill=(25, 35, 50))

        # Project particles along blade direction
        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)
        n = len(ice_np)
        step = max(1, n // 30000)
        max_pen = max(float(pen_np.max()), 0.001)

        for i in range(0, n, step):
            px, py, pz = ice_np[i]
            # Project onto blade axis
            dx = px - bx
            dy = py - by
            along = dx * cos_b + dy * sin_b
            perp = abs(-dx * sin_b + dy * cos_b)

            # Only show particles within lateral distance
            if perp < ICE_W * 0.6:
                sx, sy = to_screen(along, pz)
                if ox <= sx <= ox + pw and oy <= sy <= oy + ph:
                    pen_val = pen_np[i]
                    if pen_val > 0:
                        ci = min(255, int(pen_val / max_pen * 255))
                        color = COLOR_LUT[ci]
                        draw.point((sx, sy), fill=color)
                    else:
                        depth_frac = max(0, min(1, pz / ICE_H))
                        gray = int(40 + 60 * depth_frac)
                        draw.point((sx, sy), fill=(gray, gray, int(gray * 1.3)))

        # Draw blade cross-section (simplified)
        half_l = BLADE_LEN / 2
        edge_z = bz - 0.75 * math.cos(lean)
        # Blade bottom edge
        x0s, y0s = to_screen(-half_l, edge_z)
        x1s, y1s = to_screen(half_l, edge_z)
        draw.line([(x0s, y0s), (x1s, y1s)], fill=BLADE_COLOR, width=2)
        # Blade body (thin rectangle)
        blade_top_z = bz + 0.75 * math.cos(lean)
        x0t, y0t = to_screen(-half_l * 0.8, blade_top_z)
        x1t, y1t = to_screen(half_l * 0.8, blade_top_z)
        draw.line([(x0s, y0s), (x0t, y0t)], fill=(150, 150, 170), width=1)
        draw.line([(x1s, y1s), (x1t, y1t)], fill=(150, 150, 170), width=1)
        draw.line([(x0t, y0t), (x1t, y1t)], fill=(150, 150, 170), width=1)

        # Penetration depth annotation
        pen_mm = physics_pen_mm = getattr(self, '_last_pen_mm', 0)
        if edge_z < ICE_H:
            pen_scaled = ICE_H - edge_z
            pen_mm = pen_scaled / SCALE * 1000
            self._last_pen_mm = pen_mm
            # Arrow showing penetration
            sx_mid, sy_top = to_screen(0, ICE_H)
            _, sy_edge = to_screen(0, edge_z)
            if sy_edge > sy_top + 5:
                draw.line([(sx_mid + 30, sy_top), (sx_mid + 30, sy_edge)],
                          fill=(255, 100, 100), width=2)
                draw.text((sx_mid + 35, (sy_top + sy_edge) // 2 - 6),
                          f"{pen_mm:.3f}mm", fill=(255, 100, 100), font=self.font_small)

    def _draw_hud(self, draw, physics, y_start):
        """Draw HUD info bar at bottom of frame."""
        y = y_start
        items = [
            f"Frame: {self.frame_count}",
            f"Mass: {physics.blade_mass:.0f}kg",
            f"Lean: {math.degrees(physics.lean):.1f}\u00b0",
            f"Pen: {physics.pen_max_mm:.3f}mm",
            f"Contact: {physics.pen_contact_count} particles",
            f"F_z: {physics.reaction_fz_real:.0f}N",
            f"Paused: {physics.physics_paused}",
        ]
        x = 10
        for item in items:
            draw.text((x, y), item, fill=TEXT_COLOR, font=self.font_small)
            x += len(item) * 8 + 15

        # Color legend
        legend_y = y + 18
        draw.text((10, legend_y), "Penetration:", fill=LABEL_COLOR, font=self.font_small)
        for i in range(100):
            ci = int(i * 2.55)
            color = COLOR_LUT[ci]
            draw.rectangle([110 + i * 2, legend_y, 111 + i * 2, legend_y + 10], fill=color)
        draw.text((110, legend_y + 12), "none", fill=LABEL_COLOR, font=self.font_small)
        draw.text((280, legend_y + 12), "deep", fill=LABEL_COLOR, font=self.font_small)

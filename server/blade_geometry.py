"""
BladeGeometry — contact-length / width / area lookup from the real STL
rocker profile and hollow-grind cross-section.

A 3-D lookup table (lean x pitch x depth) is pre-computed at init so
that runtime queries are fast trilinear interpolations.
"""
import math
import time
import numpy as np

from .config import HOLLOW_RADIUS_DEFAULT


class BladeGeometry:
    """Rocker-profile + hollow-grind contact model.

    1. Extract the rocker curve (X -> rise) from the blade bottom edge.
    2. Model the hollow-grind cross-section as a circular arc.
    3. For a given (lean, pitch, depth), intersect the rocker curve with
       the ice plane to get contact_length, and compute effective_width
       from the hollow geometry.
    """

    N_PROFILE = 500   # samples along blade length
    N_CROSS   = 500   # samples across hollow-grind arc

    # ── init ──────────────────────────────────────────────────────

    def __init__(self, stl_path, hollow_radius=HOLLOW_RADIUS_DEFAULT):
        import trimesh as _tm
        from scipy.interpolate import interp1d

        m = _tm.load(stl_path)
        v = m.vertices  # STL: X=along, Y=height, Z=thickness

        # 1. Extract rocker profile (X -> rise from lowest point)
        blade_mask = (np.abs(v[:, 2]) <= 0.002) & (v[:, 1] < 0.005)
        bv = v[blade_mask]

        x_min_edge = bv[:, 0].min() + 0.002
        x_max_edge = bv[:, 0].max() - 0.002
        x_samples  = np.linspace(x_min_edge, x_max_edge, self.N_PROFILE)
        raw_y      = np.full(self.N_PROFILE, np.inf)
        for i, x in enumerate(x_samples):
            near = np.abs(bv[:, 0] - x) < 0.0005
            if near.sum() > 0:
                raw_y[i] = bv[near, 1].min()

        valid  = raw_y < np.inf
        spline = interp1d(x_samples[valid], raw_y[valid],
                          kind='cubic', fill_value='extrapolate')
        rocker_y = spline(x_samples)
        self.rocker_x    = x_samples
        self.rocker_rise = rocker_y - rocker_y.min()
        print(f"[blade_geom] Rocker profile: {valid.sum()} pts, "
              f"X=[{x_samples[0]*1000:.1f}, {x_samples[-1]*1000:.1f}]mm, "
              f"rise=[0, {self.rocker_rise.max()*1000:.2f}]mm")

        # 2. Hollow-grind cross-section model
        self.R_hollow     = hollow_radius / 1000.0
        self.blade_half_w = 0.0015

        # 3. Build lookup table
        self.lean_min,  self.lean_max,  self.lean_step  = 0.0, 61.0, 1.0
        self.pitch_min, self.pitch_max, self.pitch_step = -5.0, 5.0, 0.5
        self.depth_min, self.depth_max, self.depth_step = 0.05, 2.0, 0.05  # mm

        n_lean  = int((self.lean_max  - self.lean_min)  / self.lean_step)  + 1
        n_pitch = int((self.pitch_max - self.pitch_min) / self.pitch_step) + 1
        n_depth = int((self.depth_max - self.depth_min) / self.depth_step) + 1

        self.tbl_length = np.zeros((n_lean, n_pitch, n_depth))
        self.tbl_width  = np.zeros((n_lean, n_pitch, n_depth))
        self.tbl_area   = np.zeros((n_lean, n_pitch, n_depth))

        t0 = time.time()
        for il in range(n_lean):
            lean_rad = math.radians(self.lean_min + il * self.lean_step)
            for ip in range(n_pitch):
                pitch_rad = math.radians(self.pitch_min + ip * self.pitch_step)
                tilted_rise  = self.rocker_rise + self.rocker_x * math.sin(pitch_rad)
                tilted_rise -= tilted_rise.min()
                for id_ in range(n_depth):
                    depth_m = (self.depth_min + id_ * self.depth_step) / 1000.0
                    in_contact = tilted_rise < depth_m
                    if in_contact.sum() >= 2:
                        cx   = self.rocker_x[in_contact]
                        clen = float(cx[-1] - cx[0])
                        w_eff = self._hollow_width(lean_rad, depth_m)
                        self.tbl_length[il, ip, id_] = clen
                        self.tbl_width [il, ip, id_] = w_eff
                        self.tbl_area  [il, ip, id_] = clen * w_eff

        dt = time.time() - t0
        print(f"[blade_geom] Lookup table: {n_lean}x{n_pitch}x{n_depth} = "
              f"{n_lean * n_pitch * n_depth} entries in {dt:.1f}s")

    # ── hollow-grind width ────────────────────────────────────────

    def _hollow_width(self, lean_rad, depth_m):
        """Width of the hollow-grind arc below the ice plane at *depth_m*."""
        R  = self.R_hollow
        hw = self.blade_half_w
        z  = np.linspace(-hw, hw, self.N_CROSS)

        y_hollow = R - np.sqrt(np.maximum(0.0, R * R - z * z))

        cos_l = math.cos(lean_rad)
        sin_l = math.sin(lean_rad)
        y_leaned = y_hollow * cos_l - z * sin_l
        y_min    = y_leaned.min()

        in_contact = (y_leaned - y_min) < depth_m
        if in_contact.sum() >= 2:
            z_c = z[in_contact]
            return float(z_c[-1] - z_c[0])
        return 0.0001

    # ── equilibrium depth solver ──────────────────────────────────

    def solve_depth(self, F_normal, H_pa, lean_deg, pitch_deg,
                    tol_mm=0.005, max_iter=20):
        """Bisection: find depth d where H * Lc(d) * w(d) = F_normal."""
        lo, hi = self.depth_min, self.depth_max
        for _ in range(max_iter):
            mid = (lo + hi) * 0.5
            clen, cwid, _ = self.query(lean_deg, pitch_deg, mid)
            resist = H_pa * clen * cwid
            if resist < F_normal:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol_mm:
                break
        return (lo + hi) * 0.5

    # ── trilinear lookup ──────────────────────────────────────────

    def query(self, lean_deg, pitch_deg, depth_mm):
        """Return (contact_length_m, effective_width_m, area_m2)."""
        lean_deg  = max(self.lean_min,  min(self.lean_max  - self.lean_step,  lean_deg))
        pitch_deg = max(self.pitch_min, min(self.pitch_max - self.pitch_step, pitch_deg))
        depth_mm  = max(self.depth_min, min(self.depth_max - self.depth_step, depth_mm))

        fl = (lean_deg  - self.lean_min)  / self.lean_step
        fp = (pitch_deg - self.pitch_min) / self.pitch_step
        fd = (depth_mm  - self.depth_min) / self.depth_step

        il = int(fl); fl -= il
        ip = int(fp); fp -= ip
        id_ = int(fd); fd -= id_

        def _interp(tbl):
            c000 = tbl[il,     ip,     id_    ]
            c100 = tbl[il + 1, ip,     id_    ]
            c010 = tbl[il,     ip + 1, id_    ]
            c110 = tbl[il + 1, ip + 1, id_    ]
            c001 = tbl[il,     ip,     id_ + 1]
            c101 = tbl[il + 1, ip,     id_ + 1]
            c011 = tbl[il,     ip + 1, id_ + 1]
            c111 = tbl[il + 1, ip + 1, id_ + 1]
            c00 = c000 * (1 - fl) + c100 * fl
            c01 = c001 * (1 - fl) + c101 * fl
            c10 = c010 * (1 - fl) + c110 * fl
            c11 = c011 * (1 - fl) + c111 * fl
            c0  = c00  * (1 - fp) + c10  * fp
            c1  = c01  * (1 - fp) + c11  * fp
            return c0 * (1 - fd) + c1 * fd

        return _interp(self.tbl_length), _interp(self.tbl_width), _interp(self.tbl_area)

    # ── lowest point calculation ─────────────────────────────────

    def get_lowest_point_offset(self, lean_rad, pitch_rad):
        """Return the Z offset from blade center to lowest point.
        
        After applying pitch (tilt along blade) and lean (roll around blade axis),
        compute how far below the blade center the lowest contact point is.
        This lets us position the blade so its lowest point sits at ice surface.
        """
        # 1. Apply pitch to rocker profile
        tilted_rise = self.rocker_rise + self.rocker_x * math.sin(pitch_rad)
        min_rise = tilted_rise.min()
        
        # 2. Find where the lowest point is along blade length
        lowest_idx = np.argmin(tilted_rise)
        lowest_x = self.rocker_x[lowest_idx]
        
        # 3. At the lowest rocker point, compute hollow-grind lowest point with lean
        R = self.R_hollow
        hw = self.blade_half_w
        z = np.linspace(-hw, hw, self.N_CROSS)
        y_hollow = R - np.sqrt(np.maximum(0.0, R * R - z * z))
        
        # Rotate hollow cross-section by lean angle
        cos_l = math.cos(lean_rad)
        sin_l = math.sin(lean_rad)
        y_leaned = y_hollow * cos_l - z * sin_l
        
        # The lowest point of the leaned hollow-grind
        hollow_drop = y_leaned.min()
        
        # Total offset from blade center to lowest point (in meters)
        # Combine rocker dip + hollow-grind dip after lean
        total_offset = min_rise + hollow_drop
        
        return total_offset, lowest_x

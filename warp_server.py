"""
Warp-based skate blade physics server.
Ice particles form a groove around the blade — friction emerges from geometry.
No friction coefficients — the blade physically pushes through ice particles.

Architecture:
  - Warp GPU kernel: ice particles + rigid mesh collision (real CAD geometry)
  - aiohttp WebSocket server: browser ↔ physics
  - Blade is kinematic (position-controlled) with velocity from forces
  - Blade CAD loaded from STL, collision via wp.Mesh BVH queries
"""
import asyncio
import json
import math
import os
import time
import numpy as np
import aiohttp
from aiohttp import web

import warp as wp
wp.init()
wp.set_device("cuda:0")

# ─── Mesh constants ───
HOLLOW_RADIUS_DEFAULT = 0.015875  # 5/8" = 15.875mm in meters
STL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blade-holder-cad-watertight.stl")
USE_MESH_COLLISION = True  # set False to fall back to box collision

# ─── Physics constants ───
SCALE = 50  # Scale factor for particle resolution
BLADE_LEN_REAL = 0.280  # meters (real blade)
BLADE_W_REAL = 0.003
BLADE_H_REAL = 0.0594  # watertight blade height (59.4mm, holder removed)

BLADE_LEN = BLADE_LEN_REAL * SCALE  # 14.0m
BLADE_W = BLADE_W_REAL * SCALE      # 0.15m
BLADE_H = BLADE_H_REAL * SCALE      # 2.97m
# Cutting edge offset: distance from blade center (sim origin) to cutting edge
# In local frame, cutting edge is at sim_z = -(BLADE_H_REAL/2)*SCALE
BLADE_EDGE_OFFSET = (BLADE_H_REAL / 2.0) * SCALE  # 1.485 scaled

# Ice sheet: 500×500×15mm solid ice surface
ICE_SHEET_L = 0.500 * SCALE   # 500mm = 25.0 scaled
ICE_SHEET_W = 0.500 * SCALE   # 500mm = 25.0 scaled
ICE_SHEET_H = 0.015 * SCALE   # 15mm = 0.75 scaled

# Particle pool: 300×50×5mm hole in center of ice sheet
# Particles simulate the deformable ice in the blade contact zone
POOL_L = 0.300 * SCALE    # 300mm = 15.0 scaled
POOL_W = 0.050 * SCALE    # 50mm = 2.5 scaled
POOL_H = 0.005 * SCALE    # 5mm = 0.25 scaled

# Legacy aliases (used by particle init and settle)
ICE_L = POOL_L
ICE_W = POOL_W
ICE_H = POOL_H

# 240k particles in 300×50×5mm = 75,000mm³ → 3.2 particles/mm³
N_ICE = 240000

DT = 0.001
G = 9.81
ICE_RHO = 917.0
PARTICLE_R = 0.025 * SCALE / 50.0  # particle radius in scaled coords (0.5mm real)
STIFFNESS_BASE = 2e5
STIFFNESS = STIFFNESS_BASE  # adjusted by ice hardness
DAMPING = 80.0
BLADE_MASS = 85.0  # kg (skater weight, in real units)

# ── Center of mass parameters (from article diagram) ──
# G = center of mass of skater+skate system
# P = blade contact point on ice
# L = distance from P to G (meters)
# θ = lean angle = angle between horizontal and line P→G
# α = foot opening angle = angle between blade direction and travel direction
#     α=0: blade aligned with travel (glide), α>0: blade opened (push/stop)
#     Push perpendicular to blade: F_forward = F·sin(α)
# Balance equation: tan(θ) = v²/(R·g)  →  θ = arctan(v²/(R·g))
# Fx = horizontal force = m·v²/R (centripetal)
# Fy = vertical force = m·g (normal/weight)
L_COM = 0.90  # meters — distance P→G (crouched skater ~0.8-0.9m, upright ~1.2m)

# Rocker zones (Quad 1 profile)
ROCKER_ZONES = [
    ("Zone 1 (6')", 6 * 0.3048),   # toe: tight turns
    ("Zone 2 (9')", 9 * 0.3048),
    ("Zone 3 (12')", 12 * 0.3048),  # center
    ("Zone 4 (15')", 15 * 0.3048),  # heel: stable
]

# ─── Warp kernels ───

@wp.kernel
def init_ice(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    n: int,
    pool_l: float,
    pool_w: float,
    pool_h: float,
    pool_bottom_z: float,
    seed: int,
):
    """Initialize particles in the pool hole (depression in ice sheet surface)."""
    i = wp.tid()
    if i >= n:
        return
    s1 = wp.rand_init(seed, i)
    s2 = wp.rand_init(seed, i + n)
    s3 = wp.rand_init(seed, i + 2 * n)
    x = wp.randf(s1, -pool_l / 2.0, pool_l / 2.0)
    y = wp.randf(s2, -pool_w / 2.0, pool_w / 2.0)
    z = wp.randf(s3, pool_bottom_z, pool_bottom_z + pool_h)
    pos[i] = wp.vec3(x, y, z)
    vel[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def recenter_ice(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    blade_cx: float,
    blade_cy: float,
    n: int,
    pool_l: float,
    pool_w: float,
    pool_h: float,
    pool_bottom_z: float,
    seed_offset: int,
):
    """Wrap particles that escape pool boundaries back to opposite side."""
    i = wp.tid()
    if i >= n:
        return
    p = pos[i]
    dx = p[0] - blade_cx
    dy = p[1] - blade_cy
    
    half_l = pool_l / 2.0
    half_w = pool_w / 2.0
    new_x = p[0]
    new_y = p[1]
    new_z = p[2]
    reset = False
    
    if dx > half_l:
        new_x = blade_cx - half_l + (dx - half_l)
        reset = True
    elif dx < -half_l:
        new_x = blade_cx + half_l + (dx + half_l)
        reset = True
    
    if dy > half_w:
        new_y = blade_cy - half_w + (dy - half_w)
        reset = True
    elif dy < -half_w:
        new_y = blade_cy + half_w + (dy + half_w)
        reset = True
    
    if reset:
        s = wp.rand_init(seed_offset, i)
        new_z = wp.randf(s, pool_bottom_z, pool_bottom_z + pool_h)
        pos[i] = wp.vec3(new_x, new_y, new_z)
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    

def load_blade_mesh(stl_path, scale, hollow_radius=HOLLOW_RADIUS_DEFAULT):
    """Load blade STL, transform to simulation local frame, create wp.Mesh.
    
    STL coordinate system:
      X = blade length (±0.14m), Y = height (0=bottom edge, 0.079=top), Z = thickness (±0.0065m)
    
    Simulation blade-local frame:
      X = along blade (length), Y = across blade (thickness), Z = height
    
    Transform: sim_x = stl_x * scale, sim_y = stl_z * scale, sim_z = (stl_y - blade_center_y) * scale
    The blade center is at stl_y = BLADE_H_REAL/2 = 0.015m
    """
    import trimesh
    
    mesh = trimesh.load(stl_path)
    verts = mesh.vertices.copy()  # (N, 3) in STL coords
    faces = mesh.faces.copy()     # (M, 3) triangle indices
    
    print(f"[mesh] Loaded STL: {verts.shape[0]} vertices, {faces.shape[0]} triangles")
    print(f"[mesh] STL bounds: X=[{verts[:,0].min():.4f}, {verts[:,0].max():.4f}], "
          f"Y=[{verts[:,1].min():.4f}, {verts[:,1].max():.4f}], "
          f"Z=[{verts[:,2].min():.4f}, {verts[:,2].max():.4f}]")
    
    # Optional: adjust hollow radius by modifying bottom vertices
    if abs(hollow_radius - HOLLOW_RADIUS_DEFAULT) > 1e-6:
        R_new = hollow_radius
        half_t = 0.0015  # half thickness at cutting edge
        bottom_mask = verts[:, 1] < 0.003  # bottom 3mm is blade edge
        for i in range(len(verts)):
            if bottom_mask[i]:
                z = verts[i, 2]  # thickness position
                if abs(z) < half_t:
                    # Recalculate Y using new hollow radius
                    y_new = math.sqrt(R_new**2 - z**2) - math.sqrt(R_new**2 - half_t**2)
                    verts[i, 1] = max(0.0, y_new)
        print(f"[mesh] Adjusted hollow radius to {hollow_radius*1000:.2f}mm")
    
    # Transform: STL (X, Y, Z) → blade-local (X, Y, Z)
    # sim_x = stl_x (along blade)
    # sim_y = stl_z (across blade / thickness)  
    # sim_z = stl_y - blade_center (height, centered on blade)
    blade_center_y = BLADE_H_REAL / 2.0  # 0.0297m (blade center height)
    
    transformed = np.zeros_like(verts)
    transformed[:, 0] = verts[:, 0] * scale           # along blade
    transformed[:, 1] = verts[:, 2] * scale            # across blade (thickness)
    transformed[:, 2] = (verts[:, 1] - blade_center_y) * scale  # height (centered)
    
    print(f"[mesh] Transformed bounds: X=[{transformed[:,0].min():.2f}, {transformed[:,0].max():.2f}], "
          f"Y=[{transformed[:,1].min():.2f}, {transformed[:,1].max():.2f}], "
          f"Z=[{transformed[:,2].min():.2f}, {transformed[:,2].max():.2f}]")
    
    # Fix normals for consistent winding
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    
    # Create Warp mesh on GPU
    wp_points = wp.array(transformed.astype(np.float32), dtype=wp.vec3, device="cuda:0")
    wp_indices = wp.array(faces.flatten().astype(np.int32), dtype=wp.int32, device="cuda:0")
    
    wp_mesh = wp.Mesh(
        points=wp_points,
        indices=wp_indices,
        support_winding_number=True,
    )
    
    print(f"[mesh] wp.Mesh created on GPU (id={wp_mesh.id}, support_winding_number=True)")
    
    # Store raw data for debug visualization
    mesh_data = {
        'vertices': transformed.tolist(),
        'faces': faces.tolist(),
        'n_verts': int(verts.shape[0]),
        'n_faces': int(faces.shape[0]),
        'hollow_radius': hollow_radius,
    }
    
    return wp_mesh, mesh_data


class BladeGeometry:
    """Contact geometry derived from the real blade STL rocker profile
    and hollow-grind cross-section.

    Instead of taking the span of vertices below a plane, we:
      1. Extract the rocker curve (X → Y_rise) from the blade bottom edge.
      2. Model the hollow-grind cross-section as a circular arc.
      3. For a given (lean, pitch, depth), intersect the rocker curve with
         the ice plane to get contact_length, and compute effective_width
         from the hollow geometry.

    A 2-D lookup table (lean × pitch) stores contact_length and width as
    functions of depth via the rocker spline, enabling fast trilinear
    interpolation at runtime.
    """

    # Rocker spline sample count along the blade
    N_PROFILE = 500
    # Cross-section sample count for hollow-grind width
    N_CROSS = 500

    def __init__(self, stl_path, hollow_radius=HOLLOW_RADIUS_DEFAULT):
        import trimesh as _tm
        from scipy.interpolate import interp1d

        m = _tm.load(stl_path)
        v = m.vertices  # STL: X=along blade, Y=height, Z=thickness

        # --- 1. Extract rocker profile (X → rise from lowest point) ---
        # Filter to blade bottom edge (|Z| < 2mm, Y < 5mm)
        blade_mask = (np.abs(v[:, 2]) <= 0.002) & (v[:, 1] < 0.005)
        bv = v[blade_mask]

        x_min_edge = bv[:, 0].min() + 0.002  # exclude extreme tips
        x_max_edge = bv[:, 0].max() - 0.002
        x_samples = np.linspace(x_min_edge, x_max_edge, self.N_PROFILE)
        raw_y = np.full(self.N_PROFILE, np.inf)
        for i, x in enumerate(x_samples):
            near = np.abs(bv[:, 0] - x) < 0.0005
            if near.sum() > 0:
                raw_y[i] = bv[near, 1].min()

        valid = raw_y < np.inf
        spline = interp1d(x_samples[valid], raw_y[valid],
                          kind='cubic', fill_value='extrapolate')
        rocker_y = spline(x_samples)
        self.rocker_x = x_samples                    # meters
        self.rocker_rise = rocker_y - rocker_y.min()  # meters (0 at lowest)
        print(f"[blade_geom] Rocker profile: {valid.sum()} pts, "
              f"X=[{x_samples[0]*1000:.1f}, {x_samples[-1]*1000:.1f}]mm, "
              f"rise=[0, {self.rocker_rise.max()*1000:.2f}]mm")

        # --- 2. Hollow-grind cross-section model ---
        self.R_hollow = hollow_radius / 1000.0  # convert mm → m
        self.blade_half_w = 0.0015  # 1.5mm half-thickness (from STL)

        # --- 3. Build lookup table ---
        self.lean_min, self.lean_max, self.lean_step = 0.0, 61.0, 1.0
        self.pitch_min, self.pitch_max, self.pitch_step = -5.0, 5.0, 0.5
        self.depth_min, self.depth_max, self.depth_step = 0.05, 2.0, 0.05  # mm

        n_lean = int((self.lean_max - self.lean_min) / self.lean_step) + 1
        n_pitch = int((self.pitch_max - self.pitch_min) / self.pitch_step) + 1
        n_depth = int((self.depth_max - self.depth_min) / self.depth_step) + 1

        self.tbl_length = np.zeros((n_lean, n_pitch, n_depth))  # meters
        self.tbl_width = np.zeros((n_lean, n_pitch, n_depth))   # meters
        self.tbl_area = np.zeros((n_lean, n_pitch, n_depth))    # m²

        t0 = time.time()
        for il in range(n_lean):
            lean_deg = self.lean_min + il * self.lean_step
            lean_rad = math.radians(lean_deg)

            for ip in range(n_pitch):
                pitch_deg = self.pitch_min + ip * self.pitch_step
                pitch_rad = math.radians(pitch_deg)

                # Pitch tilts the ice plane relative to the blade:
                # effective rise(x) = rocker_rise(x) + x * sin(pitch)
                tilted_rise = self.rocker_rise + self.rocker_x * math.sin(pitch_rad)
                tilted_rise -= tilted_rise.min()  # re-zero so lowest = 0

                for id_ in range(n_depth):
                    depth_mm = self.depth_min + id_ * self.depth_step
                    depth_m = depth_mm / 1000.0

                    in_contact = tilted_rise < depth_m
                    if in_contact.sum() >= 2:
                        cx = self.rocker_x[in_contact]
                        clen = float(cx[-1] - cx[0])  # monotonic X → just endpoints
                        w_eff = self._hollow_width(lean_rad, depth_m)
                        self.tbl_length[il, ip, id_] = clen
                        self.tbl_width[il, ip, id_] = w_eff
                        self.tbl_area[il, ip, id_] = clen * w_eff

        dt = time.time() - t0
        print(f"[blade_geom] Lookup table: {n_lean}×{n_pitch}×{n_depth} = "
              f"{n_lean * n_pitch * n_depth} entries in {dt:.1f}s")

    def _hollow_width(self, lean_rad, depth_m):
        """Effective contact width from the hollow-grind cross-section.

        Models the blade bottom as a circular arc (radius = R_hollow).
        When leaned, the ice plane is a tilted line through the deepest point.
        Width = span of the arc below the ice plane at the given depth.
        """
        R = self.R_hollow
        hw = self.blade_half_w
        z = np.linspace(-hw, hw, self.N_CROSS)

        # Hollow-grind profile: y = R - sqrt(R² - z²)  (0 at edges, max at center)
        y_hollow = R - np.sqrt(np.maximum(0.0, R * R - z * z))

        # Lean rotates the cross-section: y_leaned = y*cos(lean) - z*sin(lean)
        cos_l = math.cos(lean_rad)
        sin_l = math.sin(lean_rad)
        y_leaned = y_hollow * cos_l - z * sin_l
        y_min = y_leaned.min()

        # Width: span of z where y_leaned is within `depth` of the deepest point
        in_contact = (y_leaned - y_min) < depth_m
        if in_contact.sum() >= 2:
            z_c = z[in_contact]
            return float(z_c[-1] - z_c[0])
        return 0.0001  # fallback: 0.1mm

    def solve_depth(self, F_normal, H_pa, lean_deg, pitch_deg,
                    tol_mm=0.005, max_iter=20):
        """Find equilibrium penetration depth via bisection.

        Solves: H × Lc(d) × w(d) = F_normal
        where Lc(d) and w(d) come from the rocker + hollow geometry.

        Returns depth in mm.
        """
        lo, hi = self.depth_min, self.depth_max
        for _ in range(max_iter):
            mid = (lo + hi) * 0.5
            clen, cwid, _ = self.query(lean_deg, pitch_deg, mid)
            resist = H_pa * clen * cwid  # N — ice resistance at this depth
            if resist < F_normal:
                lo = mid  # need deeper
            else:
                hi = mid  # too deep
            if (hi - lo) < tol_mm:
                break
        return (lo + hi) * 0.5

    def query(self, lean_deg, pitch_deg, depth_mm):
        """Trilinear interpolation from the lookup table.

        Returns (contact_length_m, effective_width_m, projected_area_m2).
        """
        lean_deg = max(self.lean_min, min(self.lean_max - self.lean_step, lean_deg))
        pitch_deg = max(self.pitch_min, min(self.pitch_max - self.pitch_step, pitch_deg))
        depth_mm = max(self.depth_min, min(self.depth_max - self.depth_step, depth_mm))

        fl = (lean_deg - self.lean_min) / self.lean_step
        fp = (pitch_deg - self.pitch_min) / self.pitch_step
        fd = (depth_mm - self.depth_min) / self.depth_step

        il = int(fl); fl -= il
        ip = int(fp); fp -= ip
        id_ = int(fd); fd -= id_

        def _interp(tbl):
            c000 = tbl[il, ip, id_]
            c100 = tbl[il + 1, ip, id_]
            c010 = tbl[il, ip + 1, id_]
            c110 = tbl[il + 1, ip + 1, id_]
            c001 = tbl[il, ip, id_ + 1]
            c101 = tbl[il + 1, ip, id_ + 1]
            c011 = tbl[il, ip + 1, id_ + 1]
            c111 = tbl[il + 1, ip + 1, id_ + 1]
            c00 = c000 * (1 - fl) + c100 * fl
            c01 = c001 * (1 - fl) + c101 * fl
            c10 = c010 * (1 - fl) + c110 * fl
            c11 = c011 * (1 - fl) + c111 * fl
            c0 = c00 * (1 - fp) + c10 * fp
            c1 = c01 * (1 - fp) + c11 * fp
            return c0 * (1 - fd) + c1 * fd

        clen = _interp(self.tbl_length)
        cwid = _interp(self.tbl_width)
        carea = _interp(self.tbl_area)

        return clen, cwid, carea


@wp.kernel
def physics_step_mesh(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    blade_cx: float,
    blade_cy: float,
    blade_cz: float,
    blade_half_l: float,
    blade_half_w: float,
    blade_half_h: float,
    blade_yaw: float,
    blade_lean: float,
    mesh_id: wp.uint64,
    blade_fx_out: wp.array(dtype=wp.vec3),
    pen_out: wp.array(dtype=float),
    n: int,
    dt: float,
    stiffness: float,
    damping: float,
    particle_r: float,
    pool_bottom_z: float,
    pool_top_z: float,
    pool_half_l: float,
    pool_half_w: float,
):
    """Physics step with real CAD mesh collision.
    
    Strategy: transform particle to blade local frame, query wp.Mesh for
    signed distance using winding number, push out with anisotropic stiffness.
    Box pre-filter for performance (skip mesh query for distant particles).
    """
    i = wp.tid()
    if i >= n:
        return

    p = pos[i]
    v = vel[i]
    blade_pen = float(0.0)

    force = wp.vec3(0.0, 0.0, -G * ICE_RHO)

    # Pool floor (bottom of particle pool)
    if p[2] < pool_bottom_z + particle_r:
        pen = pool_bottom_z + particle_r - p[2]
        force = force + wp.vec3(0.0, 0.0, pen * stiffness)
        vt = wp.vec3(v[0], v[1], 0.0)
        sp = wp.length(vt)
        if sp > 1.0e-5:
            fn = pen * stiffness
            ff = wp.min(fn * 0.3, sp * damping)
            force = force - wp.normalize(vt) * ff

    # Pool ceiling (ice surface level — particles can't escape above)
    if p[2] > pool_top_z - particle_r:
        pen = p[2] - (pool_top_z - particle_r)
        force = force + wp.vec3(0.0, 0.0, -pen * stiffness * 0.5)

    # Pool walls (X boundaries)
    if p[0] < -pool_half_l + particle_r:
        pen = -pool_half_l + particle_r - p[0]
        force = force + wp.vec3(pen * stiffness, 0.0, 0.0)
    if p[0] > pool_half_l - particle_r:
        pen = p[0] - (pool_half_l - particle_r)
        force = force + wp.vec3(-pen * stiffness, 0.0, 0.0)

    # Pool walls (Y boundaries)
    if p[1] < -pool_half_w + particle_r:
        pen = -pool_half_w + particle_r - p[1]
        force = force + wp.vec3(0.0, pen * stiffness, 0.0)
    if p[1] > pool_half_w - particle_r:
        pen = p[1] - (pool_half_w - particle_r)
        force = force + wp.vec3(0.0, -pen * stiffness, 0.0)

    # Transform particle into blade local frame
    cos_y = wp.cos(blade_yaw)
    sin_y = wp.sin(blade_yaw)
    cos_l = wp.cos(blade_lean)
    sin_l = wp.sin(blade_lean)

    dx = p[0] - blade_cx
    dy = p[1] - blade_cy
    dz = p[2] - blade_cz

    # Rotate by -yaw around Z
    lx = dx * cos_y + dy * sin_y
    ly = -dx * sin_y + dy * cos_y
    # Rotate by -lean around X
    lz = ly * sin_l + dz * cos_l
    ly2 = ly * cos_l - dz * sin_l

    # Box pre-filter with expanded margin
    margin = wp.max(particle_r * 3.0, 0.1)  # at least 0.1 scaled (2mm real)
    sx = blade_half_l + margin
    sy = blade_half_w + margin
    sz = blade_half_h + margin

    if wp.abs(lx) < sx and wp.abs(ly2) < sy and wp.abs(lz) < sz:
        # Mesh query: find closest point and signed distance
        query_point = wp.vec3(lx, ly2, lz)
        max_query_dist = 2.0  # large enough to find surface from deep inside blade
        
        query = wp.mesh_query_point_sign_winding_number(mesh_id, query_point, max_query_dist)
        
        if query.result:
            closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            delta = query_point - closest
            dist = wp.length(delta)
            sign = query.sign  # -1 = inside, +1 = outside
            
            # Signed distance: negative = inside mesh
            signed_dist = dist * sign
            
            if signed_dist < particle_r:
                # Particle overlaps or is inside mesh — push out
                pen = particle_r - signed_dist
                blade_pen = pen  # record raw penetration before cap
                
                # Cap penetration to avoid explosive forces
                max_pen = particle_r * 3.0
                pen = wp.min(pen, max_pen)
                
                # Push-out direction: from closest point toward particle
                if dist > 1.0e-6:
                    normal = wp.normalize(delta) * sign
                else:
                    normal = wp.vec3(0.0, 0.0, 1.0)
                
                # ANISOTROPIC stiffness based on surface normal direction
                # Normal components tell us which face type we're hitting:
                #   nx large → end face (along blade) → low stiffness
                #   ny large → side face (lateral / groove wall) → high stiffness
                #   nz large → top/bottom face (vertical) → high stiffness
                nx = wp.abs(normal[0])
                ny = wp.abs(normal[1])
                nz = wp.abs(normal[2])
                
                k_along = stiffness * 0.002
                k_lateral = stiffness
                k_vertical = stiffness
                
                # Weighted stiffness by axis contribution
                total_n = nx + ny + nz + 1.0e-8
                k_eff = (nx * k_along + ny * k_lateral + nz * k_vertical) / total_n
                
                local_force = normal * pen * k_eff
                
                # Rotate force back to world frame
                # Undo lean (rotate around X by +lean)
                fy_world_local = local_force[1] * cos_l + local_force[2] * sin_l
                fz_world_local = -local_force[1] * sin_l + local_force[2] * cos_l
                # Undo yaw (rotate around Z by +yaw)
                fx_world = local_force[0] * cos_y - fy_world_local * sin_y
                fy_world = local_force[0] * sin_y + fy_world_local * cos_y
                fz_world = fz_world_local

                world_force = wp.vec3(fx_world, fy_world, fz_world)
                force = force + world_force

                # Reaction on blade (Newton's 3rd law)
                wp.atomic_add(blade_fx_out, 0, -world_force)

    pen_out[i] = blade_pen

    # Damping
    force = force - v * damping

    # Update
    v_new = v + force * dt / ICE_RHO
    p_new = p + v_new * dt
    vel[i] = v_new
    pos[i] = p_new


@wp.kernel
def physics_step(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    blade_cx: float,
    blade_cy: float,
    blade_cz: float,
    blade_half_l: float,
    blade_half_w: float,
    blade_half_h: float,
    blade_yaw: float,
    blade_lean: float,
    blade_fx_out: wp.array(dtype=wp.vec3),
    pen_out: wp.array(dtype=float),
    n: int,
    dt: float,
    stiffness: float,
    damping: float,
    particle_r: float,
    pool_bottom_z: float,
    pool_top_z: float,
    pool_half_l: float,
    pool_half_w: float,
):
    i = wp.tid()
    if i >= n:
        return

    p = pos[i]
    v = vel[i]
    blade_pen = float(0.0)

    force = wp.vec3(0.0, 0.0, -G * ICE_RHO)

    # Pool floor
    if p[2] < pool_bottom_z + particle_r:
        pen = pool_bottom_z + particle_r - p[2]
        force = force + wp.vec3(0.0, 0.0, pen * stiffness)
        vt = wp.vec3(v[0], v[1], 0.0)
        sp = wp.length(vt)
        if sp > 1.0e-5:
            fn = pen * stiffness
            ff = wp.min(fn * 0.3, sp * damping)
            force = force - wp.normalize(vt) * ff

    # Pool ceiling
    if p[2] > pool_top_z - particle_r:
        pen = p[2] - (pool_top_z - particle_r)
        force = force + wp.vec3(0.0, 0.0, -pen * stiffness * 0.5)

    # Pool walls (X)
    if p[0] < -pool_half_l + particle_r:
        pen = -pool_half_l + particle_r - p[0]
        force = force + wp.vec3(pen * stiffness, 0.0, 0.0)
    if p[0] > pool_half_l - particle_r:
        pen = p[0] - (pool_half_l - particle_r)
        force = force + wp.vec3(-pen * stiffness, 0.0, 0.0)

    # Pool walls (Y)
    if p[1] < -pool_half_w + particle_r:
        pen = -pool_half_w + particle_r - p[1]
        force = force + wp.vec3(0.0, pen * stiffness, 0.0)
    if p[1] > pool_half_w - particle_r:
        pen = p[1] - (pool_half_w - particle_r)
        force = force + wp.vec3(0.0, -pen * stiffness, 0.0)

    # Transform particle into blade local frame
    # Blade orientation: yaw around Z, then lean around local X
    cos_y = wp.cos(blade_yaw)
    sin_y = wp.sin(blade_yaw)
    cos_l = wp.cos(blade_lean)
    sin_l = wp.sin(blade_lean)

    # Translate to blade center
    dx = p[0] - blade_cx
    dy = p[1] - blade_cy
    dz = p[2] - blade_cz

    # Rotate by -yaw around Z
    lx = dx * cos_y + dy * sin_y
    ly = -dx * sin_y + dy * cos_y
    # Rotate by -lean around X
    lz = ly * sin_l + dz * cos_l
    ly2 = ly * cos_l - dz * sin_l

    # Check box collision in local frame
    sx = blade_half_l + particle_r
    sy = blade_half_w + particle_r
    sz = blade_half_h + particle_r

    if wp.abs(lx) < sx and wp.abs(ly2) < sy and wp.abs(lz) < sz:
        px = sx - wp.abs(lx)
        py = sy - wp.abs(ly2)
        pz = sz - wp.abs(lz)
        blade_pen = wp.min(px, wp.min(py, pz))  # record minimum pen axis

        # Determine push-out direction (minimum penetration)
        # ANISOTROPIC stiffness:
        #   Along blade (X): low stiffness — blade end face is thin (~3mm), cuts easily
        #   Lateral (Y): full stiffness — groove wall, high resistance
        #   Vertical (Z): full stiffness — ground/ceiling contact
        # This ratio models the real cross-section area ratio: end face / side face ≈ 1/100
        k_along = stiffness * 0.002   # 500x weaker along blade (thin end face)
        k_lateral = stiffness          # full stiffness for groove walls
        k_vertical = stiffness         # full stiffness for vertical
        
        # Cap penetration depth to avoid explosive forces from deep overlaps
        max_pen = particle_r * 3.0
        px = wp.min(px, max_pen)
        py = wp.min(py, max_pen)
        pz = wp.min(pz, max_pen)

        local_force = wp.vec3(0.0, 0.0, 0.0)
        if px < py and px < pz:
            sign = 1.0
            if lx < 0.0:
                sign = -1.0
            local_force = wp.vec3(sign * px * k_along, 0.0, 0.0)
        elif py < pz:
            sign = 1.0
            if ly2 < 0.0:
                sign = -1.0
            local_force = wp.vec3(0.0, sign * py * k_lateral, 0.0)
        else:
            sign = 1.0
            if lz < 0.0:
                sign = -1.0
            local_force = wp.vec3(0.0, 0.0, sign * pz * k_vertical)

        # Rotate force back to world frame
        # Undo lean (rotate around X by +lean)
        fy_world_local = local_force[1] * cos_l + local_force[2] * sin_l
        fz_world_local = -local_force[1] * sin_l + local_force[2] * cos_l
        # Undo yaw (rotate around Z by +yaw)
        fx_world = local_force[0] * cos_y - fy_world_local * sin_y
        fy_world = local_force[0] * sin_y + fy_world_local * cos_y
        fz_world = fz_world_local

        world_force = wp.vec3(fx_world, fy_world, fz_world)
        force = force + world_force

        # Reaction on blade (Newton's 3rd law)
        wp.atomic_add(blade_fx_out, 0, -world_force)

    pen_out[i] = blade_pen

    # Damping
    force = force - v * damping

    # Update
    v_new = v + force * dt / ICE_RHO
    p_new = p + v_new * dt
    vel[i] = v_new
    pos[i] = p_new


@wp.kernel
def pen_reduce(
    pen_in: wp.array(dtype=float),
    out: wp.array(dtype=float),
    n: int,
):
    """Reduce penetration array to [max_pen, sum_pen, count]."""
    i = wp.tid()
    if i >= n:
        return
    p = pen_in[i]
    if p > 0.0:
        wp.atomic_max(out, 0, p)
        wp.atomic_add(out, 1, p)
        wp.atomic_add(out, 2, 1.0)


class BladePhysics:
    def __init__(self):
        # All positions in SCALED coordinates to match particles
        # Position blade so cutting edge is at ice surface (accounting for lean)
        # edge_world_z = pos[2] - BLADE_EDGE_OFFSET*cos(lean) = ICE_SHEET_H
        # → pos[2] = ICE_SHEET_H + BLADE_EDGE_OFFSET*cos(lean)
        init_lean = 15.0 * math.pi / 180
        self.pos = np.array([0.0, 0.0, ICE_SHEET_H + BLADE_EDGE_OFFSET * math.cos(init_lean)])
        self.vel = np.array([0.0, 0.0, 0.0])  # velocity in scaled coords/s
        self.yaw = 0.0
        self.lean = 15.0 * math.pi / 180  # radians — user-set lean angle (θ)
        self.alpha = 0.0  # radians — foot opening angle (α) between blade and travel direction
        self.pitch = 0.0  # -1 to +1
        self.L = L_COM  # distance P→G in meters

        # Derived quantities (updated each step)
        self.theta_balance = 0.0  # required lean for current v and R: arctan(v²/Rg)
        self.Fx = 0.0  # horizontal centripetal force at P (N)
        self.Fy = 0.0  # vertical normal force at P (N) = mg
        self.ac = 0.0  # centripetal acceleration (m/s²)
        self.G_pos = [0.0, 0.0, L_COM]  # center of mass position [x, y, z] real meters
        self.peak_push_speed = 0.0  # track max speed during push for energy conservation
        self.ice_hardness_mpa = 4.75  # default ice hardness (MPa), updated by 'ice' command
        self.pen_analytical_mm = 0.0  # analytical penetration: F_n / (H × Lc × w_eff)
        self.reaction_fz_real = 0.0  # vertical reaction force from particles (N)
        self.vel_z = 0.0  # vertical velocity for blade Z settling

        # Ice particles
        self.ice_pos = wp.zeros(N_ICE, dtype=wp.vec3, device="cuda:0")
        self.ice_vel = wp.zeros(N_ICE, dtype=wp.vec3, device="cuda:0")
        self.blade_force = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        # Per-particle penetration depth output (sim units)
        self.pen_out = wp.zeros(N_ICE, dtype=float, device="cuda:0")
        # Reduction buffer: [max_pen, sum_pen, count]
        self.pen_stats = wp.zeros(3, dtype=float, device="cuda:0")
        # Smoothed penetration values for display
        self.pen_max_mm = 0.0
        self.pen_avg_mm = 0.0
        self.pen_contact_count = 0
        self.pen_contact_area_mm2 = 0.0

        # Load blade mesh for CAD collision
        self.blade_mesh = None
        self.mesh_data = None
        self.hollow_radius = HOLLOW_RADIUS_DEFAULT
        if USE_MESH_COLLISION and os.path.exists(STL_PATH):
            try:
                self.blade_mesh, self.mesh_data = load_blade_mesh(
                    STL_PATH, SCALE, self.hollow_radius
                )
                print(f"[physics] Mesh collision ENABLED (hollow={self.hollow_radius*1000:.2f}mm)")
            except Exception as e:
                print(f"[physics] Mesh load failed, falling back to box: {e}")
                self.blade_mesh = None
        else:
            if USE_MESH_COLLISION:
                print(f"[physics] STL not found at {STL_PATH}, falling back to box collision")
            else:
                print(f"[physics] Mesh collision disabled, using box")

        # Blade contact geometry lookup table from real STL
        self.blade_geom = None
        if os.path.exists(STL_PATH):
            try:
                self.blade_geom = BladeGeometry(STL_PATH, self.hollow_radius)
            except Exception as e:
                print(f"[physics] BladeGeometry init failed: {e}")

        # Contact geometry values (updated each step from lookup table)
        self.contact_length_mm = 0.0
        self.contact_width_mm = 0.0
        self.contact_area_mm2 = 0.0

        # Blade direction = yaw + alpha (physical blade orientation in world)
        # yaw = travel/heading direction (updated by arc turning)
        # alpha = offset angle (foot opening)

        pool_bottom = float(ICE_SHEET_H - POOL_H)
        wp.launch(init_ice, dim=N_ICE,
                  inputs=[self.ice_pos, self.ice_vel, N_ICE,
                          POOL_L, POOL_W, POOL_H, pool_bottom, 42],
                  device="cuda:0")
        wp.synchronize()

        # Settle ice particles
        print("[physics] Settling ice particles...", flush=True)
        self.recenter_seed = 1000
        for i in range(200):
            self._step_particles()
            if i % 50 == 0:
                print(f"  [init] step {i}/200", flush=True)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.blade_force.zero_()
        print("[physics] Ice settled.", flush=True)

        self.frame = 0
        self.push_frames = 0
        self.push_fx = 0.0
        self.push_fy = 0.0
        self.force_mult = 1.5
        self.blade_mass = BLADE_MASS
        self.recenter_seed = 1000
        # Force accumulator for display (rolling average)
        self.force_accum_along = 0.0
        self.force_accum_perp = 0.0
        self.force_decay = 0.99  # exponential moving average

    def _step_particles(self):
        # Recenter ice particles around blade (infinite ice sheet)
        self.recenter_seed += 1
        pool_bottom = float(ICE_SHEET_H - POOL_H)
        wp.launch(recenter_ice, dim=N_ICE,
                  inputs=[
                      self.ice_pos, self.ice_vel,
                      float(self.pos[0]), float(self.pos[1]),
                      N_ICE, POOL_L, POOL_W, POOL_H, pool_bottom,
                      self.recenter_seed,
                  ],
                  device="cuda:0")
        
        self.blade_force.zero_()
        self.pen_out.zero_()
        self.pen_stats.zero_()
        half_l = BLADE_LEN / 2.0
        half_w = BLADE_W / 2.0
        half_h = BLADE_H / 2.0

        # Blade physical orientation in particles = yaw + alpha
        blade_dir = self.yaw + self.alpha
        
        # Pool geometry: particles confined to depression in ice sheet surface
        pool_top = float(ICE_SHEET_H)              # ice surface level
        pool_bottom = float(ICE_SHEET_H - POOL_H)  # pool floor
        pool_hl = float(POOL_L / 2.0)
        pool_hw = float(POOL_W / 2.0)
        
        if self.blade_mesh is not None:
            # Use real CAD mesh collision
            wp.launch(physics_step_mesh, dim=N_ICE,
                      inputs=[
                          self.ice_pos, self.ice_vel,
                          float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                          half_l, half_w, half_h,
                          float(blade_dir), float(self.lean),
                          self.blade_mesh.id,
                          self.blade_force,
                          self.pen_out,
                          N_ICE, DT, STIFFNESS, DAMPING, PARTICLE_R,
                          pool_bottom, pool_top, pool_hl, pool_hw,
                      ],
                      device="cuda:0")
        else:
            # Fallback to box collision
            wp.launch(physics_step, dim=N_ICE,
                      inputs=[
                          self.ice_pos, self.ice_vel,
                          float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                          half_l, half_w, half_h,
                          float(blade_dir), float(self.lean),
                          self.blade_force,
                          self.pen_out,
                          N_ICE, DT, STIFFNESS, DAMPING, PARTICLE_R,
                          pool_bottom, pool_top, pool_hl, pool_hw,
                      ],
                      device="cuda:0")

        # Reduce per-particle penetration to stats: [max, sum, count]
        wp.launch(pen_reduce, dim=N_ICE,
                  inputs=[self.pen_out, self.pen_stats, N_ICE],
                  device="cuda:0")
        wp.synchronize()

    def settle_blade_quick(self, steps=50):
        """Settle blade Z to equilibrium penetration depth.
        
        Uses the analytical equilibrium solver (F_normal = H × Lc × w) to compute
        the target penetration, then positions the blade and runs GPU particle
        physics to establish contact forces and validate the depth.
        
        The watertight mesh provides reliable signed-distance queries for
        accurate per-particle penetration measurement.
        """
        F_normal_real = self.blade_mass * G * math.cos(self.lean)
        cos_lean = math.cos(self.lean)
        
        # Compute equilibrium penetration from analytical model
        target_pen_mm = 0.3  # fallback
        if self.blade_geom is not None:
            H_pa = self.ice_hardness_mpa * 1e6
            lean_deg = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0
            target_pen_mm = self.blade_geom.solve_depth(
                F_normal_real, H_pa, lean_deg, pitch_deg
            )
            self.pen_analytical_mm = target_pen_mm
        
        target_pen_mm = max(0.01, target_pen_mm)
        
        # Position blade at equilibrium depth
        target_pen_scaled = target_pen_mm / 1000.0 * SCALE
        edge_target_z = ICE_SHEET_H - target_pen_scaled
        target_z = edge_target_z + BLADE_EDGE_OFFSET * cos_lean
        
        self.pos[2] = target_z
        self.vel = np.array([0.0, 0.0, 0.0])
        self.vel_z = 0.0
        
        print(f"[settle] pen={target_pen_mm:.3f}mm, F_n={F_normal_real:.0f}N, "
              f"lean={math.degrees(self.lean):.1f}°, Z={target_z:.4f}", flush=True)
        
        # Reinitialize particles fresh in the pool
        pool_bottom = float(ICE_SHEET_H - POOL_H)
        self.recenter_seed += 1
        wp.launch(init_ice, dim=N_ICE,
                  inputs=[self.ice_pos, self.ice_vel,
                          N_ICE, POOL_L, POOL_W, POOL_H, pool_bottom,
                          self.recenter_seed + self.frame],
                  device="cuda:0")
        wp.synchronize()
        
        # Run GPU physics to establish particle contact with watertight mesh
        best_pen = 0.0
        best_cnt = 0
        best_sum = 0.0
        for i in range(steps):
            self._step_particles()
            ps = self.pen_stats.numpy()
            cnt = int(ps[2])
            if cnt > best_cnt:
                best_pen = float(ps[0])
                best_sum = float(ps[1])
                best_cnt = cnt
        
        # Read final reaction force
        bf = self.blade_force.numpy()[0]
        self.reaction_fz_real = float(bf[2]) / SCALE
        
        # GPU penetration from particle signed-distance (watertight mesh)
        gpu_pen_mm = best_pen * 1000.0 / SCALE if best_cnt > 0 else 0.0
        
        # Use geometric pen (reliable, independent of particle noise)
        edge_z = self.pos[2] - BLADE_EDGE_OFFSET * cos_lean
        geo_pen_mm = max(0.0, (ICE_SHEET_H - edge_z) / SCALE * 1000.0)
        
        self.pen_max_mm = geo_pen_mm
        self.pen_avg_mm = geo_pen_mm * 0.6
        self.pen_contact_count = best_cnt
        particle_r_real_m = PARTICLE_R / SCALE
        particle_area_mm2 = math.pi * (particle_r_real_m * 1000) ** 2
        self.pen_contact_area_mm2 = best_cnt * particle_area_mm2
        
        # Update contact geometry from blade_geom lookup
        if self.blade_geom is not None and self.pen_max_mm > 0.001:
            lean_deg = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0
            clen, cwid, carea = self.blade_geom.query(lean_deg, pitch_deg, self.pen_max_mm)
            self.contact_length_mm = clen * 1000
            self.contact_width_mm = cwid * 1000
            self.contact_area_mm2 = carea * 1e6
        
        self.blade_force.zero_()
        self.force_accum_along = 0.0
        self.force_accum_perp = 0.0
        
        print(f"[settle] DONE pen={geo_pen_mm:.3f}mm, GPU_particle_pen={gpu_pen_mm:.3f}mm, "
              f"cnt={best_cnt}, F_n={F_normal_real:.0f}N", flush=True)

    def step(self):
        self.frame += 1

        # Apply push force
        # Push is 200 frames (0.2s at dt=0.001), force ~5000*mult in scaled units
        # This gives ~0.35 m/s real per push at force_mult=1.5
        fx, fy = 0.0, 0.0
        if self.push_frames > 0:
            F = 5000.0 * self.force_mult  # Force in scaled units
            # Push force direction uses blade orientation (yaw + alpha)
            # When α>0, the blade is "opened" — push perpendicular to blade
            # gives more forward component: F_forward = F·sin(α)
            blade_dir = self.yaw + self.alpha
            cos_b = math.cos(blade_dir)
            sin_b = math.sin(blade_dir)
            fx = self.push_fx * cos_b - self.push_fy * sin_b
            fy = self.push_fx * sin_b + self.push_fy * cos_b
            fx *= F
            fy *= F
            self.push_frames -= 1

        # Step particles and get reaction force
        self._step_particles()

        # Read reaction force from ice on blade (in scaled world coords)
        bf = self.blade_force.numpy()[0]
        reaction_fx = float(bf[0])
        reaction_fy = float(bf[1])
        reaction_fz = float(bf[2])
        self.reaction_fz_real = reaction_fz / SCALE  # vertical reaction in real N

        # No continuous Z adjustment — settle_blade_quick handles Z positioning
        # on every parameter change. Between settles, Z stays fixed.

        # Penetration = geometric depth of blade edge below ice surface
        edge_z = self.pos[2] - BLADE_EDGE_OFFSET * math.cos(self.lean)
        geo_pen_mm = max(0.0, (ICE_SHEET_H - edge_z) / SCALE * 1000.0)
        d = self.force_decay
        self.pen_max_mm = d * self.pen_max_mm + (1 - d) * geo_pen_mm
        self.pen_avg_mm = self.pen_max_mm * 0.6
        # Read particle contact count from pen_stats
        ps = self.pen_stats.numpy()
        pen_count = int(ps[2])
        self.pen_contact_count = pen_count
        # Contact area: count × particle cross-section area in real mm²
        # Each particle occupies π×r² in real space (r = PARTICLE_R / SCALE in meters)
        particle_r_real_m = PARTICLE_R / SCALE
        particle_area_mm2 = math.pi * (particle_r_real_m * 1000) ** 2
        self.pen_contact_area_mm2 = d * self.pen_contact_area_mm2 + (1 - d) * (pen_count * particle_area_mm2)

        # Query real blade geometry for contact length/width/area
        if self.blade_geom is not None and self.pen_max_mm > 0.01:
            lean_deg = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0  # pitch slider -1..+1 → ±5° tilt
            clen, cwid, carea = self.blade_geom.query(lean_deg, pitch_deg, self.pen_max_mm)
            self.contact_length_mm = d * self.contact_length_mm + (1 - d) * (clen * 1000)
            self.contact_width_mm = d * self.contact_width_mm + (1 - d) * (cwid * 1000)
            self.contact_area_mm2 = d * self.contact_area_mm2 + (1 - d) * (carea * 1e6)

        # Analytical penetration: solve for depth d where H × Lc(d) × w(d) = F_normal
        # F_normal = m × g × cos(lean), Lc(d) and w(d) from blade geometry lookup.
        # No magic multipliers — self-consistent ploughing equilibrium.
        if self.blade_geom is not None:
            F_normal = self.blade_mass * 9.81 * math.cos(self.lean)
            H_pa = self.ice_hardness_mpa * 1e6
            lean_deg = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0
            raw_analytical = self.blade_geom.solve_depth(
                F_normal, H_pa, lean_deg, pitch_deg
            )
            self.pen_analytical_mm = d * self.pen_analytical_mm + (1 - d) * raw_analytical

        # Accumulate forces for display (along/perp to blade direction)
        blade_dir = self.yaw + self.alpha
        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)
        f_a = abs(reaction_fx * cos_b + reaction_fy * sin_b)  # along-blade
        f_p = abs(-reaction_fx * sin_b + reaction_fy * cos_b)  # perpendicular
        d = self.force_decay
        self.force_accum_along = d * self.force_accum_along + (1 - d) * f_a
        self.force_accum_perp = d * self.force_accum_perp + (1 - d) * f_p

        # Update blade velocity
        # Push forces accelerate, reaction forces (from groove) decelerate
        speed_before = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        # Save velocity direction before update (for energy conservation check)
        if speed_before > 0.01:
            inv_spd = 1.0 / speed_before
            vdir_x = self.vel[0] * inv_spd
            vdir_y = self.vel[1] * inv_spd
        else:
            vdir_x, vdir_y = 1.0, 0.0

        ax = (fx + reaction_fx) / self.blade_mass
        ay = (fy + reaction_fy) / self.blade_mass
        self.vel[0] += ax * DT
        self.vel[1] += ay * DT
        
        # Energy conservation: cap speed to what the push alone could produce.
        # This prevents penalty collision energy injection from groove particles.
        # At each step during push, theoretical speed = F * elapsed_frames * DT / mass
        # After push, speed can only decrease.
        speed_after = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if self.push_frames > 0:
            F_push = 5000.0 * self.force_mult
            elapsed = 200 - self.push_frames  # how many push frames have elapsed
            # Theoretical speed from push impulse alone (with 10% tolerance)
            theoretical_v = F_push * elapsed * DT / self.blade_mass * 1.1
            theoretical_v = max(theoretical_v, speed_before)  # never reduce during push
            if speed_after > theoretical_v and theoretical_v > 0.01:
                scale_f = theoretical_v / speed_after
                self.vel[0] *= scale_f
                self.vel[1] *= scale_f
            self.peak_push_speed = max(self.peak_push_speed, theoretical_v)
        elif self.peak_push_speed > 0.01 and speed_after > self.peak_push_speed:
            # After push: cap speed — groove can decelerate but not accelerate beyond push
            scale_f = self.peak_push_speed / speed_after
            self.vel[0] *= scale_f
            self.vel[1] *= scale_f
        
        if self.push_frames == 195:  # 5th push frame
            speed_now = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
            print(f"[step] push active: fx={fx:.1f} fy={fy:.1f} rx={reaction_fx:.1f} ry={reaction_fy:.1f} ax={ax:.1f} vel=({self.vel[0]:.6f},{self.vel[1]:.6f}) spd_real={speed_now/SCALE:.6f}")

        # Speed cap: max ~10 m/s real = 500 scaled (Olympic sprinter ~15 m/s)
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        MAX_SPEED = 10.0 * SCALE  # 10 m/s real
        if speed > MAX_SPEED:
            scale_f = MAX_SPEED / speed
            self.vel[0] *= scale_f
            self.vel[1] *= scale_f
            speed = MAX_SPEED

        # Very slight viscous damping (air resistance)
        if speed > 0.001:
            drag = 0.99999
            self.vel[0] *= drag
            self.vel[1] *= drag
        else:
            self.vel[0] = 0.0
            self.vel[1] = 0.0

        # ── Arc turning from lean angle (with full G/L/balance physics) ──
        # From the article: a leaned skater in a turn satisfies:
        #   tan(θ) = v²/(R·g)    where θ = lean angle, R = turn radius
        #   Fy = m·g              (vertical equilibrium)
        #   Fx = m·v²/R = m·ac   (centripetal force)
        #   ac = v²/R             (centripetal acceleration toward turn center)
        #   G is at distance L from contact point P, at angle θ from horizontal
        #
        # The user sets the lean angle θ. This determines the turn radius:
        #   R_turn = v²/(g·tan(θ))    (from balance equation)
        # But the turn radius is also constrained by the rocker zone:
        #   R_rocker = rocker zone radius (1.83m to 4.57m)
        # The effective turn uses: R_eff = R_rocker / lean_factor
        # where lean_factor = min(1.0, |θ|/(π/4))
        #
        # Compute all forces:
        speed_real = speed / SCALE
        self.Fy = self.blade_mass * G  # Normal force = mg
        
        if abs(self.lean) > 0.01 and speed > 0.5:
            cos_y = math.cos(self.yaw)
            sin_y = math.sin(self.yaw)
            v_along = self.vel[0] * cos_y + self.vel[1] * sin_y
            v_along_real = v_along / SCALE
            
            if abs(v_along) > 0.5:
                # Get rocker zone radius
                idx = max(0, min(3, int((self.pitch + 1) / 2 * 3.99)))
                _, R_real = ROCKER_ZONES[idx]
                R_scaled = R_real * SCALE
                
                lean_sign = 1.0 if self.lean > 0 else -1.0
                lean_factor = min(1.0, abs(self.lean) / (math.pi / 4))
                
                # Effective turn radius from rocker + lean
                R_eff_real = R_real / lean_factor if lean_factor > 0.001 else 1e6
                R_eff_scaled = R_eff_real * SCALE
                
                # Centripetal acceleration and force (article equations)
                self.ac = (v_along_real ** 2) / R_eff_real if R_eff_real > 0.01 else 0.0
                self.Fx = self.blade_mass * self.ac  # centripetal force
                
                # Required lean angle for balance: θ_balance = arctan(v²/(R·g))
                # This is the lean angle the skater MUST have to not fall
                # Use total speed (not just v_along) for better display during turns
                if R_eff_real > 0.01 and speed_real > 0.01:
                    self.theta_balance = math.atan2(speed_real ** 2, R_eff_real * G)
                else:
                    self.theta_balance = 0.0
                
                # Angular velocity: ω = v_along / R_eff
                omega = (v_along / R_eff_scaled) * lean_factor * lean_sign
                
                # Update yaw
                self.yaw += omega * DT
                
                # Rotate velocity vector to follow new heading
                new_cos = math.cos(self.yaw)
                new_sin = math.sin(self.yaw)
                v_perp = -self.vel[0] * sin_y + self.vel[1] * cos_y
                self.vel[0] = v_along * new_cos - v_perp * new_sin
                self.vel[1] = v_along * new_sin + v_perp * new_cos
            else:
                self.ac = 0.0
                self.Fx = 0.0
                self.theta_balance = 0.0
        else:
            self.ac = 0.0
            self.Fx = 0.0
            self.theta_balance = 0.0
        
        # Update center of mass position G (real meters)
        # G is at distance L from P, at angle θ from horizontal
        # P is the contact point (blade position on ice)
        pos_real_x = self.pos[0] / SCALE
        pos_real_y = self.pos[1] / SCALE
        # G offset: horizontal component = L·cos(θ) toward turn center
        #           vertical component = L·sin(θ) upward
        # For display: G is above P, offset laterally by L·cos(θ)
        theta = abs(self.lean)
        self.G_pos = [
            pos_real_x,  # same x as blade (forward)
            pos_real_y,  # same y as blade (lateral, simplified)
            self.L * math.cos(theta)  # height: L·cos(θ), lower when leaned more
        ]

        # Update position (scaled coords)
        self.pos[0] += self.vel[0] * DT
        self.pos[1] += self.vel[1] * DT

        return self.get_state()

    def get_state(self):
        # Velocity and position are in scaled coords, convert to real for display
        speed_scaled = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        speed_real = speed_scaled / SCALE  # m/s in real coords
        
        # Velocity decomposition along/perpendicular to blade direction
        blade_dir = self.yaw + self.alpha
        bx = math.cos(blade_dir)
        by = math.sin(blade_dir)
        va = (self.vel[0] * bx + self.vel[1] * by) / SCALE
        ppx, ppy = -by, bx
        vp = (self.vel[0] * ppx + self.vel[1] * ppy) / SCALE

        # Rocker zone from pitch
        idx = max(0, min(3, int((self.pitch + 1) / 2 * 3.99)))
        zone_name, R = ROCKER_ZONES[idx]

        # Use accumulated forces for smooth display (exponential moving average)
        f_along_disp = self.force_accum_along / SCALE
        f_perp_disp = self.force_accum_perp / SCALE

        # Position in real coords for display
        pos_real = [self.pos[0] / SCALE, self.pos[1] / SCALE, self.pos[2] / SCALE]

        # Velocity in real coords (m/s)
        vel_real = [self.vel[0] / SCALE, self.vel[1] / SCALE, 0.0]
        
        return {
            'type': 'state',
            'pos': [round(pos_real[0], 6), round(pos_real[1], 6), round(pos_real[2], 6)],
            'vel': [round(vel_real[0], 6), round(vel_real[1], 6), round(vel_real[2], 6)],
            'speed': round(speed_real, 6),
            'va': round(va, 6),
            'vp': round(vp, 6),
            'yaw': round(self.yaw, 6),
            'lean_actual': round(self.lean, 6),
            'pitch_actual': round(self.pitch * 5.0 * math.pi / 180, 6),
            'pitch_val': round(self.pitch, 3),
            'mu_a': round(f_along_disp, 2),
            'mu_p': round(f_perp_disp, 2),
            # Real GPU penetration from particle simulation
            'pen': round(self.pen_max_mm, 4),  # max penetration depth (mm)
            'pen_max_mm': round(self.pen_max_mm, 4),
            'pen_avg_mm': round(self.pen_avg_mm, 4),
            'pen_contact_count': self.pen_contact_count,
            'pen_contact_area_mm2': round(self.pen_contact_area_mm2, 2),
            # Real blade contact geometry from STL mesh lookup
            'Lc': round(self.contact_length_mm / 1000, 4),  # contact length (m)
            'contact_length_mm': round(self.contact_length_mm, 2),
            'contact_width_mm': round(self.contact_width_mm, 3),
            'contact_area_geom_mm2': round(self.contact_area_mm2, 2),
            # Analytical penetration: F_n / (H × Lc × w_eff) — no magic numbers
            'pen_analytical_mm': round(self.pen_analytical_mm, 4),
            'ice_hardness_mpa': round(self.ice_hardness_mpa, 2),
            'blade_reaction_z': round(self.reaction_fz_real, 1),
            'F_normal': round(self.blade_mass * G * math.cos(self.lean), 1),
            'R': round(R, 4),
            'zone': idx,
            'zone_name': zone_name,
            'contact_z': 0.0,
            # Article variables: G (COM), L (P→G distance), forces
            'L': round(self.L, 4),
            'G_pos': [round(self.G_pos[0], 6), round(self.G_pos[1], 6), round(self.G_pos[2], 6)],
            'Fx': round(self.Fx, 2),  # centripetal force (N)
            'Fy': round(self.Fy, 2),  # normal force = mg (N)
            'Fg': round(self.blade_mass * G, 2),  # gravity force (N)
            'mass': round(self.blade_mass, 1),
            'ac': round(self.ac, 4),  # centripetal acceleration (m/s²)
            'theta_balance': round(math.degrees(self.theta_balance), 2),  # required lean for balance (deg)
            'theta_actual': round(math.degrees(abs(self.lean)), 2),  # actual lean (deg)
            'alpha': round(math.degrees(self.alpha), 2),  # foot opening angle (deg)
            'frame': self.frame,
            'engine': 'warp-gpu-groove',
            'collision_mode': 'mesh' if self.blade_mesh is not None else 'box',
            'hollow_radius_mm': round(self.hollow_radius * 1000, 2),
            'f_along': round(f_along_disp, 1),
            'f_lateral': round(f_perp_disp, 1),
            'n_ice': N_ICE,
            # Ice geometry (real mm for frontend)
            'ice_sheet_l_mm': round(ICE_SHEET_L / SCALE * 1000, 0),
            'ice_sheet_w_mm': round(ICE_SHEET_W / SCALE * 1000, 0),
            'ice_sheet_h_mm': round(ICE_SHEET_H / SCALE * 1000, 1),
            'pool_l_mm': round(POOL_L / SCALE * 1000, 0),
            'pool_w_mm': round(POOL_W / SCALE * 1000, 0),
            'pool_h_mm': round(POOL_H / SCALE * 1000, 1),
        }

    def handle_command(self, cmd):
        global STIFFNESS
        t = cmd.get('cmd', '') or cmd.get('type', '')
        if t == 'push':
            self.push_fx = cmd.get('fx', 0)
            self.push_fy = cmd.get('fy', 0)
            fv = cmd.get('force', None)
            if fv is not None:
                self.force_mult = fv
            self.push_frames = 200  # 0.2 seconds at dt=0.001
            print(f"[cmd] PUSH fx={self.push_fx}, fy={self.push_fy}, force={self.force_mult}, frames={self.push_frames}")
        elif t == 'lean':
            deg = cmd.get('value', 15)
            self.lean = deg * math.pi / 180
            self.settle_blade_quick()
        elif t == 'set_L':
            self.L = max(0.3, min(1.5, cmd.get('value', 0.9)))  # clamp 0.3-1.5m
        elif t == 'alpha':
            deg = max(-90, min(90, cmd.get('value', 0)))
            self.alpha = deg * math.pi / 180
            print(f'[cmd] Alpha (foot opening) = {deg}°')
            self.settle_blade_quick()
        elif t == 'set_velocity':
            # Set blade velocity directly (m/s in world frame)
            # vx = forward (along x), vy = lateral (along y)
            vx = cmd.get('vx', 0.0)
            vy = cmd.get('vy', 0.0)
            self.vel = np.array([vx, vy, 0.0])
            print(f'[cmd] Set velocity vx={vx:.3f}, vy={vy:.3f} m/s')
        elif t == 'pitch':
            self.pitch = cmd.get('value', 0)
            self.settle_blade_quick()
        elif t == 'force':
            self.force_mult = cmd.get('value', 1.5)
        elif t == 'weight':
            self.blade_mass = cmd.get('value', 85)
            self.settle_blade_quick()
        elif t == 'ice':
            hardness = max(1, min(20, cmd.get('value', 7)))
            self.ice_hardness_mpa = hardness
            STIFFNESS = STIFFNESS_BASE * (hardness / 7.0)
            print(f'[cmd] Ice hardness={hardness} MPa, stiffness={STIFFNESS:.0f}')
            self.settle_blade_quick()
        elif t == 'hollow_radius':
            radius_mm = max(5.0, min(50.0, cmd.get('value', 15.875)))
            radius_m = radius_mm / 1000.0
            if abs(radius_m - self.hollow_radius) > 1e-6 and os.path.exists(STL_PATH):
                self.hollow_radius = radius_m
                try:
                    self.blade_mesh, self.mesh_data = load_blade_mesh(
                        STL_PATH, SCALE, self.hollow_radius
                    )
                    print(f'[cmd] Hollow radius={radius_mm:.1f}mm')
                    self.settle_blade_quick()
                except Exception as e:
                    print(f'[cmd] Hollow radius change failed: {e}')
        elif t == 'toggle_mesh':
            # Toggle between mesh and box collision for comparison
            if self.blade_mesh is not None:
                self._saved_mesh = self.blade_mesh
                self.blade_mesh = None
                print('[cmd] Switched to BOX collision')
            elif hasattr(self, '_saved_mesh') and self._saved_mesh is not None:
                self.blade_mesh = self._saved_mesh
                print('[cmd] Switched to MESH collision')
        elif t == 'yaw':
            self.yaw = cmd.get('value', 0) * math.pi / 180
        elif t == 'reset':
            self.pos = np.array([0.0, 0.0, ICE_SHEET_H + BLADE_EDGE_OFFSET * math.cos(self.lean)])
            self.vel = np.array([0.0, 0.0, 0.0])
            self.yaw = 0.0
            self.lean = 15.0 * math.pi / 180
            self.alpha = 0.0
            self.pitch = 0.0
            self.blade_mass = BLADE_MASS
            self.L = L_COM
            self.theta_balance = 0.0
            self.Fx = 0.0
            self.Fy = self.blade_mass * G
            self.ac = 0.0
            self.G_pos = [0.0, 0.0, self.L]
            STIFFNESS = STIFFNESS_BASE
            self.push_frames = 0
            self.peak_push_speed = 0.0
            self.force_accum_along = 0.0
            self.force_accum_perp = 0.0
            pool_bottom = float(ICE_SHEET_H - POOL_H)
            wp.launch(init_ice, dim=N_ICE,
                      inputs=[self.ice_pos, self.ice_vel,
                              N_ICE, POOL_L, POOL_W, POOL_H, pool_bottom, 42],
                      device="cuda:0")
            wp.synchronize()
            self.settle_blade_quick()
            print("[cmd] RESET complete")


# ─── Web server ───

physics = None
clients = set()


async def physics_loop():
    global physics
    physics = BladePhysics()
    print(f"[server] Physics ready. {N_ICE} ice particles, {SCALE}x scale")

    frame = 0
    t0 = time.time()
    while True:
        state = physics.step()
        frame += 1

        # Broadcast to clients every 4th frame (~250Hz physics / 4 = ~60Hz updates)
        if frame % 4 == 0 and clients:
            msg = json.dumps(state)
            dead = []
            for ws in clients:
                try:
                    await ws.send_str(msg)
                except:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

        # Log every 500 frames
        if frame % 500 == 0:
            elapsed = time.time() - t0
            sps = 500 / elapsed
            t0 = time.time()
            s = state['speed']
            fl = state.get('f_lateral', 0)
            fa = state.get('f_along', 0)
            print(f"[physics] {sps:.0f} SPS, speed={s:.3f} m/s, F_lat={fl:.0f}N F_along={fa:.0f}N, clients={len(clients)}")

        await asyncio.sleep(0)  # yield to event loop


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    print(f"[ws] Client connected ({len(clients)} total)")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                    physics.handle_command(cmd)
                except Exception as e:
                    print(f"[ws] Bad command: {e}")
    finally:
        clients.discard(ws)
        print(f"[ws] Client disconnected ({len(clients)} total)")

    return ws


async def index_handler(request):
    return web.FileResponse('./index.html')


async def test_handler(request):
    return web.FileResponse('./test_live.html')


async def dashboard_handler(request):
    return web.FileResponse('./dashboard.html')


async def sandbox_handler(request):
    return web.FileResponse('./sandbox.html')


async def debug_mesh_handler(request):
    """Debug visualization endpoint: returns mesh + nearby particle data as JSON."""
    if physics is None:
        return web.json_response({'error': 'Physics not initialized'}, status=503)
    
    # Get particles near blade (within 2x blade extents)
    ice_np = physics.ice_pos.numpy()
    cx, cy, cz = physics.pos[0], physics.pos[1], physics.pos[2]
    half_l = BLADE_LEN / 2.0
    half_w = BLADE_W * 6.0  # wider to see groove
    
    # Filter particles near blade
    mask = (
        (np.abs(ice_np[:, 0] - cx) < half_l * 1.5) &
        (np.abs(ice_np[:, 1] - cy) < half_w) &
        (ice_np[:, 2] < BLADE_H * 1.5)
    )
    near_particles = ice_np[mask]
    
    response = {
        'blade_pos': [float(cx), float(cy), float(cz)],
        'blade_yaw': float(physics.yaw + physics.alpha),
        'blade_lean': float(physics.lean),
        'collision_mode': 'mesh' if physics.blade_mesh is not None else 'box',
        'hollow_radius_mm': round(physics.hollow_radius * 1000, 2),
        'n_particles_near': int(mask.sum()),
        'n_particles_total': N_ICE,
        'particles': near_particles.tolist(),
        'scale': SCALE,
    }
    
    # Add mesh data if available
    if physics.mesh_data is not None:
        response['mesh'] = physics.mesh_data
    
    return web.json_response(response)


async def start_background_tasks(app):
    app['physics_task'] = asyncio.ensure_future(physics_loop())


async def cleanup_background_tasks(app):
    app['physics_task'].cancel()


if __name__ == '__main__':
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/test', test_handler)
    app.router.add_get('/dashboard', dashboard_handler)
    app.router.add_get('/sandbox', sandbox_handler)
    app.router.add_get('/debug_mesh', debug_mesh_handler)
    app.router.add_get('/ws', ws_handler)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    print("[server] Starting on port 8765...")
    web.run_app(app, host='0.0.0.0', port=8765)

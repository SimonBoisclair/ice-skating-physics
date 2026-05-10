"""
Warp GPU kernels for ice-particle simulation.

All kernels run on CUDA. They are compiled once at import time by Warp's JIT.
"""
import warp as wp

from .config import G, ICE_RHO


@wp.func
def dem_contact_force(n: wp.vec3, v: wp.vec3, gap: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    vn = wp.dot(n, v)
    fn = wp.max(-gap * k_n - vn * k_d, 0.0)
    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 1.0e-6:
        vt = vt / vs

    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))
    return n * fn - vt * ft


@wp.func
def closest_point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


# ────────────────────────────────────────────────────────────────────
# Particle initialisation
# ────────────────────────────────────────────────────────────────────

@wp.kernel
def init_ice(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    n: int,
    ice_l: float,
    ice_w: float,
    ice_h: float,
    seed: int,
):
    i = wp.tid()
    if i >= n:
        return
    s1 = wp.rand_init(seed, i)
    s2 = wp.rand_init(seed, i + n)
    s3 = wp.rand_init(seed, i + 2 * n)
    x = wp.randf(s1, -ice_l / 2.0, ice_l / 2.0)
    y = wp.randf(s2, -ice_w / 2.0, ice_w / 2.0)
    z = wp.randf(s3, 0.0, ice_h)
    pos[i] = wp.vec3(x, y, z)
    vel[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def init_ice_lattice(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    n: int,
    ice_l: float,
    ice_w: float,
    ice_h: float,
    nx: int,
    ny: int,
    seed: int,
):
    i = wp.tid()
    if i >= n:
        return

    layer = nx * ny
    iz = i / layer
    rem = i - iz * layer
    iy = rem / nx
    ix = rem - iy * nx

    sx = ice_l / float(nx)
    sy = ice_w / float(ny)
    nz = (n + layer - 1) / layer
    sz = ice_h / float(nz)

    s1 = wp.rand_init(seed, i)
    s2 = wp.rand_init(seed, i + n)
    s3 = wp.rand_init(seed, i + 2 * n)
    jitter_x = wp.randf(s1, -0.2, 0.2) * sx
    jitter_y = wp.randf(s2, -0.2, 0.2) * sy
    jitter_z = wp.randf(s3, -0.2, 0.2) * sz

    x = -ice_l * 0.5 + (float(ix) + 0.5) * sx + jitter_x
    y = -ice_w * 0.5 + (float(iy) + 0.5) * sy + jitter_y
    z = (float(iz) + 0.5) * sz + jitter_z
    pos[i] = wp.vec3(x, y, z)
    vel[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def physics_step_particles_only(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pen: wp.array(dtype=float),
    cube_pos: wp.array(dtype=wp.vec3),
    cube_vel: wp.array(dtype=wp.vec3),
    cube_force: wp.array(dtype=float),
    cube_verts: wp.array(dtype=wp.vec3),
    cube_faces: wp.array(dtype=int),
    cube_face_count: int,
    n: int,
    dt: float,
    stiffness: float,
    particle_r: float,
    ice_l: float,
    ice_w: float,
    ice_h: float,
    cube_size: float,
    cube_contact_stiffness: float,
    cube_contact_damping: float,
    cube_contact_friction: float,
    contact_damping: float,
):
    tid = wp.tid()
    if tid >= n:
        return

    i = wp.hash_grid_point_id(grid, tid)
    if i >= n:
        return

    p = pos[i]
    v = vel[i]
    force = wp.vec3(0.0, 0.0, -G * ICE_RHO)
    contact_d = 2.0 * particle_r
    k_contact = stiffness
    k_damp = contact_damping
    k_friction = stiffness * 0.1
    k_mu = 0.5
    max_pen = 0.0

    if p[2] < particle_r:
        floor_pen = particle_r - p[2]
        max_pen = wp.max(max_pen, floor_pen)
        force = force + dem_contact_force(wp.vec3(0.0, 0.0, 1.0), v, -floor_pen, k_contact, k_damp, k_friction, k_mu)

    if p[0] < -ice_l * 0.5 + particle_r:
        wall_pen = (-ice_l * 0.5 + particle_r) - p[0]
        max_pen = wp.max(max_pen, wall_pen)
        force = force + dem_contact_force(wp.vec3(1.0, 0.0, 0.0), v, -wall_pen, k_contact, k_damp, k_friction, k_mu)
    if p[0] > ice_l * 0.5 - particle_r:
        wall_pen = p[0] - (ice_l * 0.5 - particle_r)
        max_pen = wp.max(max_pen, wall_pen)
        force = force + dem_contact_force(wp.vec3(-1.0, 0.0, 0.0), v, -wall_pen, k_contact, k_damp, k_friction, k_mu)
    if p[1] < -ice_w * 0.5 + particle_r:
        wall_pen = (-ice_w * 0.5 + particle_r) - p[1]
        max_pen = wp.max(max_pen, wall_pen)
        force = force + dem_contact_force(wp.vec3(0.0, 1.0, 0.0), v, -wall_pen, k_contact, k_damp, k_friction, k_mu)
    if p[1] > ice_w * 0.5 - particle_r:
        wall_pen = p[1] - (ice_w * 0.5 - particle_r)
        max_pen = wp.max(max_pen, wall_pen)
        force = force + dem_contact_force(wp.vec3(0.0, -1.0, 0.0), v, -wall_pen, k_contact, k_damp, k_friction, k_mu)

    c = cube_pos[0]
    cv = cube_vel[0]
    cube_contact_d = particle_r
    closest_dist = cube_contact_d
    closest_normal = wp.vec3(0.0, 0.0, 1.0)
    for f in range(cube_face_count):
        ia = cube_faces[f * 3 + 0]
        ib = cube_faces[f * 3 + 1]
        ic = cube_faces[f * 3 + 2]
        a = cube_verts[ia] + c
        b = cube_verts[ib] + c
        tri_c = cube_verts[ic] + c
        q = closest_point_triangle(p, a, b, tri_c)
        delta_cube = p - q
        dist_cube = wp.length(delta_cube)
        if dist_cube < closest_dist:
            if dist_cube > 1.0e-6:
                closest_normal = delta_cube / dist_cube
            else:
                tri_n = wp.cross(b - a, tri_c - a)
                tri_n_len = wp.length(tri_n)
                if tri_n_len > 1.0e-6:
                    closest_normal = tri_n / tri_n_len
            closest_dist = dist_cube

    if closest_dist < cube_contact_d:
        cube_pen = cube_contact_d - closest_dist
        contact_force = dem_contact_force(closest_normal, v - cv, -cube_pen, cube_contact_stiffness, cube_contact_damping, cube_contact_friction, k_mu)
        max_pen = wp.max(max_pen, cube_pen)
        force = force + contact_force
        wp.atomic_add(cube_force, 0, -contact_force[0])
        wp.atomic_add(cube_force, 1, -contact_force[1])
        wp.atomic_add(cube_force, 2, -contact_force[2])

    neighbors = wp.hash_grid_query(grid, p, contact_d * 1.05)
    for j in neighbors:
        if j != i:
            delta = p - pos[j]
            dist = wp.length(delta)
            if dist > 1.0e-6 and dist < contact_d:
                normal = delta / dist
                vrel = v - vel[j]
                gap = dist - contact_d
                max_pen = wp.max(max_pen, -gap)
                force = force + dem_contact_force(normal, vrel, gap, k_contact, k_damp, k_friction, k_mu)

    v_new = v + force * dt / ICE_RHO
    p_new = p + v_new * dt
    p_new = wp.vec3(
        wp.clamp(p_new[0], -ice_l * 0.5 + particle_r, ice_l * 0.5 - particle_r),
        wp.clamp(p_new[1], -ice_w * 0.5 + particle_r, ice_w * 0.5 - particle_r),
        wp.clamp(p_new[2], particle_r, ice_h - particle_r),
    )
    vel[i] = v_new
    pos[i] = p_new
    pen[i] = max_pen


# ────────────────────────────────────────────────────────────────────
# Reduction kernel (per-particle pen → [max, sum, count])
# ────────────────────────────────────────────────────────────────────

@wp.kernel
def pen_reduce(
    pen_in: wp.array(dtype=float),
    out: wp.array(dtype=float),
    n: int,
):
    i = wp.tid()
    if i >= n:
        return
    p = pen_in[i]
    if p > 0.0:
        wp.atomic_max(out, 0, p)
        wp.atomic_add(out, 1, p)
        wp.atomic_add(out, 2, 1.0)

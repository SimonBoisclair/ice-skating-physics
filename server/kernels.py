"""
Warp GPU kernels for ice-particle simulation.

All kernels run on CUDA. They are compiled once at import time by Warp's
JIT and then launched from BladePhysics._step_particles().
"""
import warp as wp

from .config import G, ICE_RHO

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
def recenter_ice(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    blade_cx: float,
    blade_cy: float,
    n: int,
    ice_l: float,
    ice_w: float,
    ice_h: float,
    seed_offset: int,
):
    """Wrap particles that stray too far from the blade back to the other side."""
    i = wp.tid()
    if i >= n:
        return
    p = pos[i]
    dx = p[0] - blade_cx
    dy = p[1] - blade_cy

    half_l = ice_l / 2.0
    half_w = ice_w / 2.0
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
        new_z = wp.randf(s, 0.0, ice_h)
        pos[i] = wp.vec3(new_x, new_y, new_z)
        vel[i] = wp.vec3(0.0, 0.0, 0.0)


# ────────────────────────────────────────────────────────────────────
# Physics step — mesh collision (real CAD geometry)
# ────────────────────────────────────────────────────────────────────

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
):
    """Transform particle → blade-local, query signed distance, push out."""
    i = wp.tid()
    if i >= n:
        return

    p = pos[i]
    v = vel[i]
    blade_pen = float(0.0)

    force = wp.vec3(0.0, 0.0, -G * ICE_RHO)

    # Ground plane
    if p[2] < particle_r:
        pen = particle_r - p[2]
        force = force + wp.vec3(0.0, 0.0, pen * stiffness)
        vt = wp.vec3(v[0], v[1], 0.0)
        sp = wp.length(vt)
        if sp > 1.0e-5:
            fn = pen * stiffness
            ff = wp.min(fn * 0.3, sp * damping)
            force = force - wp.normalize(vt) * ff

    # Transform to blade-local frame
    cos_y = wp.cos(blade_yaw)
    sin_y = wp.sin(blade_yaw)
    cos_l = wp.cos(blade_lean)
    sin_l = wp.sin(blade_lean)

    dx = p[0] - blade_cx
    dy = p[1] - blade_cy
    dz = p[2] - blade_cz

    lx  =  dx * cos_y + dy * sin_y
    ly  = -dx * sin_y + dy * cos_y
    lz  =  ly * sin_l + dz * cos_l
    ly2 =  ly * cos_l - dz * sin_l

    # Box pre-filter
    margin = wp.max(particle_r * 3.0, 0.1)
    sx = blade_half_l + margin
    sy = blade_half_w + margin
    sz = blade_half_h + margin

    if wp.abs(lx) < sx and wp.abs(ly2) < sy and wp.abs(lz) < sz:
        query_point = wp.vec3(lx, ly2, lz)
        max_query_dist = 2.0

        query = wp.mesh_query_point_sign_winding_number(mesh_id, query_point, max_query_dist)

        if query.result:
            closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            delta = query_point - closest
            dist = wp.length(delta)
            sign = query.sign

            signed_dist = dist * sign

            if signed_dist < particle_r:
                pen = particle_r - signed_dist
                blade_pen = pen

                max_pen = particle_r * 3.0
                pen = wp.min(pen, max_pen)

                if dist > 1.0e-6:
                    normal = wp.normalize(delta) * sign
                else:
                    normal = wp.vec3(0.0, 0.0, 1.0)

                # Anisotropic stiffness by surface-normal direction
                nx = wp.abs(normal[0])
                ny = wp.abs(normal[1])
                nz = wp.abs(normal[2])

                k_along   = stiffness * 0.002
                k_lateral = stiffness
                k_vertical = stiffness

                total_n = nx + ny + nz + 1.0e-8
                k_eff = (nx * k_along + ny * k_lateral + nz * k_vertical) / total_n

                local_force = normal * pen * k_eff

                # Rotate back to world frame
                fy_world_local = local_force[1] * cos_l + local_force[2] * sin_l
                fz_world_local = -local_force[1] * sin_l + local_force[2] * cos_l
                fx_world = local_force[0] * cos_y - fy_world_local * sin_y
                fy_world = local_force[0] * sin_y + fy_world_local * cos_y
                fz_world = fz_world_local

                world_force = wp.vec3(fx_world, fy_world, fz_world)
                force = force + world_force
                wp.atomic_add(blade_fx_out, 0, -world_force)

    pen_out[i] = blade_pen

    # Damping + integration
    force = force - v * damping
    v_new = v + force * dt / ICE_RHO
    p_new = p + v_new * dt
    vel[i] = v_new
    pos[i] = p_new


# ────────────────────────────────────────────────────────────────────
# Physics step — box collision (fallback when mesh unavailable)
# ────────────────────────────────────────────────────────────────────

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
):
    i = wp.tid()
    if i >= n:
        return

    p = pos[i]
    v = vel[i]
    blade_pen = float(0.0)

    force = wp.vec3(0.0, 0.0, -G * ICE_RHO)

    # Ground plane
    if p[2] < particle_r:
        pen = particle_r - p[2]
        force = force + wp.vec3(0.0, 0.0, pen * stiffness)
        vt = wp.vec3(v[0], v[1], 0.0)
        sp = wp.length(vt)
        if sp > 1.0e-5:
            fn = pen * stiffness
            ff = wp.min(fn * 0.3, sp * damping)
            force = force - wp.normalize(vt) * ff

    # Transform to blade-local frame
    cos_y = wp.cos(blade_yaw)
    sin_y = wp.sin(blade_yaw)
    cos_l = wp.cos(blade_lean)
    sin_l = wp.sin(blade_lean)

    dx = p[0] - blade_cx
    dy = p[1] - blade_cy
    dz = p[2] - blade_cz

    lx  =  dx * cos_y + dy * sin_y
    ly  = -dx * sin_y + dy * cos_y
    lz  =  ly * sin_l + dz * cos_l
    ly2 =  ly * cos_l - dz * sin_l

    sx = blade_half_l + particle_r
    sy = blade_half_w + particle_r
    sz = blade_half_h + particle_r

    if wp.abs(lx) < sx and wp.abs(ly2) < sy and wp.abs(lz) < sz:
        px = sx - wp.abs(lx)
        py = sy - wp.abs(ly2)
        pz = sz - wp.abs(lz)
        blade_pen = wp.min(px, wp.min(py, pz))

        k_along   = stiffness * 0.002
        k_lateral = stiffness
        k_vertical = stiffness

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

        # Rotate back to world frame
        fy_world_local = local_force[1] * cos_l + local_force[2] * sin_l
        fz_world_local = -local_force[1] * sin_l + local_force[2] * cos_l
        fx_world = local_force[0] * cos_y - fy_world_local * sin_y
        fy_world = local_force[0] * sin_y + fy_world_local * cos_y
        fz_world = fz_world_local

        world_force = wp.vec3(fx_world, fy_world, fz_world)
        force = force + world_force
        wp.atomic_add(blade_fx_out, 0, -world_force)

    pen_out[i] = blade_pen

    # Damping + integration
    force = force - v * damping
    v_new = v + force * dt / ICE_RHO
    p_new = p + v_new * dt
    vel[i] = v_new
    pos[i] = p_new


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

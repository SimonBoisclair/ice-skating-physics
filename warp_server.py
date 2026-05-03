"""
Warp-based skate blade physics server.
Ice particles form a groove around the blade — friction emerges from geometry.
No friction coefficients — the blade physically pushes through ice particles.

Architecture:
  - Warp GPU kernel: ice particles + rigid blade collision
  - aiohttp WebSocket server: browser ↔ physics
  - Blade is kinematic (position-controlled) with velocity from forces
"""
import asyncio
import json
import math
import time
import numpy as np
import aiohttp
from aiohttp import web

import warp as wp
wp.init()
wp.set_device("cuda:0")

# ─── Physics constants ───
SCALE = 50  # Scale factor for particle resolution
BLADE_LEN_REAL = 0.280  # meters (real blade)
BLADE_W_REAL = 0.003
BLADE_H_REAL = 0.030

BLADE_LEN = BLADE_LEN_REAL * SCALE  # 14.0m
BLADE_W = BLADE_W_REAL * SCALE      # 0.15m
BLADE_H = BLADE_H_REAL * SCALE      # 1.5m

ICE_L = BLADE_LEN * 0.5  # 7m patch
ICE_W = BLADE_W * 12      # 1.8m wide (35% clearance for 45° lean blade footprint of 1.17m)
ICE_H = BLADE_H * 0.4     # 0.6m deep
# Maintain same particle density as original (2381/m³): 7 × 1.8 × 0.6 = 7.56 m³ × 2381 ≈ 18000
N_ICE = 18000

DT = 0.001
G = 9.81
ICE_RHO = 917.0
PARTICLE_R = 0.025 * SCALE / 50.0  # scale with sim
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
    """Wrap particles that are too far from blade back to other side (infinite ice sheet)."""
    i = wp.tid()
    if i >= n:
        return
    p = pos[i]
    dx = p[0] - blade_cx
    dy = p[1] - blade_cy
    
    # If particle is outside ice region around blade, wrap to opposite side
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
        # Reset wrapped particle to fresh ice at random height
        s = wp.rand_init(seed_offset, i)
        new_z = wp.randf(s, 0.0, ice_h)
        pos[i] = wp.vec3(new_x, new_y, new_z)
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    

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

    # Damping
    force = force - v * damping

    # Update
    v_new = v + force * dt / ICE_RHO
    p_new = p + v_new * dt
    vel[i] = v_new
    pos[i] = p_new


class BladePhysics:
    def __init__(self):
        # All positions in SCALED coordinates to match particles
        self.pos = np.array([0.0, 0.0, ICE_H * 0.5])  # blade center in ice field
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

        # Ice particles
        self.ice_pos = wp.zeros(N_ICE, dtype=wp.vec3, device="cuda:0")
        self.ice_vel = wp.zeros(N_ICE, dtype=wp.vec3, device="cuda:0")
        self.blade_force = wp.zeros(1, dtype=wp.vec3, device="cuda:0")

        # Blade direction = yaw + alpha (physical blade orientation in world)
        # yaw = travel/heading direction (updated by arc turning)
        # alpha = offset angle (foot opening)

        wp.launch(init_ice, dim=N_ICE,
                  inputs=[self.ice_pos, self.ice_vel, N_ICE,
                          ICE_L, ICE_W, ICE_H, 42],
                  device="cuda:0")
        wp.synchronize()

        # Settle ice (blade is fixed during settling)
        print("[physics] Settling ice particles...")
        self.recenter_seed = 1000
        for _ in range(500):
            self._step_particles()
        # Zero out any accumulated blade velocity + force from settling
        self.vel = np.array([0.0, 0.0, 0.0])
        self.blade_force.zero_()
        print("[physics] Ice settled.")

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
        wp.launch(recenter_ice, dim=N_ICE,
                  inputs=[
                      self.ice_pos, self.ice_vel,
                      float(self.pos[0]), float(self.pos[1]),
                      N_ICE, ICE_L, ICE_W, ICE_H,
                      self.recenter_seed,
                  ],
                  device="cuda:0")
        
        self.blade_force.zero_()
        half_l = BLADE_LEN / 2.0
        half_w = BLADE_W / 2.0
        half_h = BLADE_H / 2.0

        # Blade physical orientation in particles = yaw + alpha
        blade_dir = self.yaw + self.alpha
        wp.launch(physics_step, dim=N_ICE,
                  inputs=[
                      self.ice_pos, self.ice_vel,
                      float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                      half_l, half_w, half_h,
                      float(blade_dir), float(self.lean),
                      self.blade_force,
                      N_ICE, DT, STIFFNESS, DAMPING, PARTICLE_R,
                  ],
                  device="cuda:0")
        wp.synchronize()

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

        return {
            'type': 'state',
            'pos': [round(pos_real[0], 6), round(pos_real[1], 6), round(pos_real[2], 6)],
            'speed': round(speed_real, 6),
            'va': round(va, 6),
            'vp': round(vp, 6),
            'yaw': round(self.yaw, 6),
            'lean_actual': round(self.lean, 6),
            'pitch_actual': round(self.pitch * 5.0 * math.pi / 180, 6),
            'pitch_val': round(self.pitch, 3),
            'mu_a': round(f_along_disp, 2),
            'mu_p': round(f_perp_disp, 2),
            'pen': round(min(2.0, 0.3 + math.degrees(self.lean) / 45.0 * 1.7), 3),  # 0.3-2.0mm based on lean
            'Lc': round(BLADE_LEN_REAL * 0.15, 4),
            'R': round(R, 4),
            'zone': idx,
            'zone_name': zone_name,
            'contact_z': 0.0,
            # Article variables: G (COM), L (P→G distance), forces
            'L': round(self.L, 4),
            'G_height': round(self.G_pos[2], 4),
            'Fx': round(self.Fx, 2),  # centripetal force (N)
            'Fy': round(self.Fy, 2),  # normal force = mg (N)
            'ac': round(self.ac, 4),  # centripetal acceleration (m/s²)
            'theta_balance': round(math.degrees(self.theta_balance), 2),  # required lean for balance (deg)
            'theta_actual': round(math.degrees(abs(self.lean)), 2),  # actual lean (deg)
            'alpha': round(math.degrees(self.alpha), 2),  # foot opening angle (deg)
            'frame': self.frame,
            'engine': 'warp-gpu-groove',
            'f_along': round(f_along_disp, 1),
            'f_lateral': round(f_perp_disp, 1),
            'n_ice': N_ICE,
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
            old_lean = self.lean
            deg = cmd.get('value', 15)
            self.lean = deg * math.pi / 180
            # Re-settle ice when lean changes significantly to avoid particle overlap explosions
            if abs(self.lean - old_lean) > 0.03:  # ~2°
                settle_steps = min(1500, int(300 + abs(self.lean - old_lean) * 800))
                self.vel = np.array([0.0, 0.0, 0.0])
                for _ in range(settle_steps):
                    self._step_particles()
                self.vel = np.array([0.0, 0.0, 0.0])
                self.blade_force.zero_()
                self.force_accum_along = 0.0
                self.force_accum_perp = 0.0
        elif t == 'set_L':
            self.L = max(0.3, min(1.5, cmd.get('value', 0.9)))  # clamp 0.3-1.5m
        elif t == 'alpha':
            # Foot opening angle: 0° (aligned) to 90° (perpendicular/hockey stop)
            deg = max(-90, min(90, cmd.get('value', 0)))
            old_alpha = self.alpha
            self.alpha = deg * math.pi / 180
            print(f'[cmd] Alpha (foot opening) = {deg}°')
            # Re-settle ice when alpha changes significantly
            if abs(self.alpha - old_alpha) > 0.05:  # ~3°
                settle_steps = min(800, int(200 + abs(self.alpha - old_alpha) * 400))
                self.vel = np.array([0.0, 0.0, 0.0])
                for _ in range(settle_steps):
                    self._step_particles()
                self.vel = np.array([0.0, 0.0, 0.0])
                self.blade_force.zero_()
                self.force_accum_along = 0.0
                self.force_accum_perp = 0.0
        elif t == 'set_velocity':
            # Set blade velocity directly (m/s in world frame)
            # vx = forward (along x), vy = lateral (along y)
            vx = cmd.get('vx', 0.0)
            vy = cmd.get('vy', 0.0)
            self.vel = np.array([vx, vy, 0.0])
            print(f'[cmd] Set velocity vx={vx:.3f}, vy={vy:.3f} m/s')
        elif t == 'pitch':
            self.pitch = cmd.get('value', 0)
        elif t == 'force':
            self.force_mult = cmd.get('value', 1.5)
        elif t == 'weight':
            self.blade_mass = cmd.get('value', 85)
        elif t == 'ice':
            # Ice hardness: soft (2 MPa) → less stiffness, hard (15 MPa) → more stiffness
            # Scale STIFFNESS proportionally: medium=7 MPa → 1x, soft=2 → 0.4x, hard=15 → 2.1x
            hardness = max(1, min(20, cmd.get('value', 7)))
            old_stiff = STIFFNESS
            STIFFNESS = STIFFNESS_BASE * (hardness / 7.0)
            # Re-settle ice with new stiffness to avoid explosive overlap
            # Higher stiffness needs more settle steps (larger forces)
            if abs(STIFFNESS - old_stiff) > 1000:
                stiff_ratio = max(1.0, STIFFNESS / STIFFNESS_BASE)
                settle = min(1000, int(300 * stiff_ratio))
                self.vel = np.array([0.0, 0.0, 0.0])
                for _ in range(settle):
                    self._step_particles()
                self.vel = np.array([0.0, 0.0, 0.0])
                self.blade_force.zero_()
                self.force_accum_along = 0.0
                self.force_accum_perp = 0.0
            print(f'[cmd] Ice hardness={hardness} MPa, stiffness={STIFFNESS:.0f}')
        elif t == 'yaw':
            self.yaw = cmd.get('value', 0) * math.pi / 180
        elif t == 'reset':
            self.pos = np.array([0.0, 0.0, ICE_H * 0.5])
            self.vel = np.array([0.0, 0.0, 0.0])
            self.yaw = 0.0
            self.lean = 15.0 * math.pi / 180  # reset to default
            self.alpha = 0.0  # reset foot opening angle
            self.pitch = 0.0
            self.blade_mass = BLADE_MASS
            self.L = L_COM
            self.theta_balance = 0.0
            self.Fx = 0.0
            self.Fy = self.blade_mass * G
            self.ac = 0.0
            self.G_pos = [0.0, 0.0, self.L]
            STIFFNESS = STIFFNESS_BASE  # reset ice hardness
            self.push_frames = 0
            self.peak_push_speed = 0.0
            self.force_accum_along = 0.0
            self.force_accum_perp = 0.0
            # Re-init ice around blade and settle
            wp.launch(init_ice, dim=N_ICE,
                      inputs=[self.ice_pos, self.ice_vel,
                              N_ICE, ICE_L, ICE_W, ICE_H, 42],
                      device="cuda:0")
            wp.synchronize()
            for _ in range(300):
                self._step_particles()
            self.vel = np.array([0.0, 0.0, 0.0])
            self.blade_force.zero_()
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


async def start_background_tasks(app):
    app['physics_task'] = asyncio.ensure_future(physics_loop())


async def cleanup_background_tasks(app):
    app['physics_task'].cancel()


if __name__ == '__main__':
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/test', test_handler)
    app.router.add_get('/dashboard', dashboard_handler)
    app.router.add_get('/ws', ws_handler)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    print("[server] Starting on port 8765...")
    web.run_app(app, host='0.0.0.0', port=8765)

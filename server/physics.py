"""
BladePhysics — main simulation class.

Owns the GPU particle arrays, blade state, and the step/settle logic.
Command dispatch for WebSocket messages lives in handle_command().
"""
import math
import os
import numpy as np
import warp as wp

from .config import (
    SCALE, BLADE_LEN, BLADE_W, BLADE_H, BLADE_H_REAL,
    ICE_L, ICE_W, ICE_H, N_ICE,
    DT, G, PARTICLE_R, STIFFNESS_BASE, DAMPING, BLADE_MASS,
    L_COM, ROCKER_ZONES, STL_PATH, USE_MESH_COLLISION,
    HOLLOW_RADIUS_DEFAULT,
)
from .kernels import init_ice, recenter_ice, physics_step_mesh, physics_step, pen_reduce
from .blade_mesh import load_blade_mesh
from .blade_geometry import BladeGeometry

# Module-level mutable stiffness (adjusted by ice-hardness commands)
STIFFNESS = STIFFNESS_BASE


class BladePhysics:
    def __init__(self):
        # Blade position in scaled coordinates
        init_lean = 15.0 * math.pi / 180
        self.pos = np.array([0.0, 0.0, ICE_H + 0.75 * math.cos(init_lean)])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.yaw   = 0.0
        self.lean   = init_lean
        self.alpha  = 0.0        # foot opening angle (rad)
        self.pitch  = 0.0        # -1..+1
        self.L      = L_COM

        # Derived display quantities
        self.theta_balance   = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.ac = 0.0
        self.G_pos = [0.0, 0.0, L_COM]
        self.peak_push_speed = 0.0
        self.ice_hardness_mpa = 4.75
        self.pen_analytical_mm = 0.0
        self.reaction_fz_real  = 0.0
        self.vel_z = 0.0

        # ── GPU arrays ──
        self.ice_pos     = wp.zeros(N_ICE, dtype=wp.vec3,  device="cuda:0")
        self.ice_vel     = wp.zeros(N_ICE, dtype=wp.vec3,  device="cuda:0")
        self.blade_force = wp.zeros(1,     dtype=wp.vec3,  device="cuda:0")
        self.pen_out     = wp.zeros(N_ICE, dtype=float,    device="cuda:0")
        self.pen_stats   = wp.zeros(3,     dtype=float,    device="cuda:0")

        # Smoothed display values
        self.pen_max_mm          = 0.0
        self.pen_avg_mm          = 0.0
        self.pen_contact_count   = 0
        self.pen_contact_area_mm2 = 0.0

        # ── Blade mesh (CAD collision) ──
        self.blade_mesh   = None
        self.mesh_data    = None
        self.hollow_radius = HOLLOW_RADIUS_DEFAULT
        if USE_MESH_COLLISION and os.path.exists(STL_PATH):
            try:
                self.blade_mesh, self.mesh_data = load_blade_mesh(
                    STL_PATH, SCALE, self.hollow_radius)
                print(f"[physics] Mesh collision ENABLED (hollow={self.hollow_radius*1000:.2f}mm)")
            except Exception as e:
                print(f"[physics] Mesh load failed, falling back to box: {e}")
        else:
            tag = "STL not found" if USE_MESH_COLLISION else "disabled"
            print(f"[physics] Mesh collision {tag}, using box")

        # ── Blade geometry lookup table ──
        self.blade_geom = None
        if os.path.exists(STL_PATH):
            try:
                self.blade_geom = BladeGeometry(STL_PATH, self.hollow_radius)
            except Exception as e:
                print(f"[physics] BladeGeometry init failed: {e}")

        self.contact_length_mm = 0.0
        self.contact_width_mm  = 0.0
        self.contact_area_mm2  = 0.0

        # ── Initialise particles ──
        wp.launch(init_ice, dim=N_ICE,
                  inputs=[self.ice_pos, self.ice_vel, N_ICE,
                          ICE_L, ICE_W, ICE_H, 42],
                  device="cuda:0")
        wp.synchronize()

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
        self.force_accum_along = 0.0
        self.force_accum_perp  = 0.0
        self.force_decay = 0.99

    # ── particle physics step ─────────────────────────────────────

    def _step_particles(self):
        """Run one physics sub-step on all ice particles."""
        self.recenter_seed += 1
        wp.launch(recenter_ice, dim=N_ICE,
                  inputs=[self.ice_pos, self.ice_vel,
                          float(self.pos[0]), float(self.pos[1]),
                          N_ICE, ICE_L, ICE_W, ICE_H,
                          self.recenter_seed],
                  device="cuda:0")

        self.blade_force.zero_()
        self.pen_out.zero_()
        self.pen_stats.zero_()
        half_l = BLADE_LEN / 2.0
        half_w = BLADE_W   / 2.0
        half_h = BLADE_H   / 2.0
        blade_dir = self.yaw + self.alpha

        if self.blade_mesh is not None:
            wp.launch(physics_step_mesh, dim=N_ICE,
                      inputs=[self.ice_pos, self.ice_vel,
                              float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                              half_l, half_w, half_h,
                              float(blade_dir), float(self.lean),
                              self.blade_mesh.id,
                              self.blade_force, self.pen_out,
                              N_ICE, DT, STIFFNESS, DAMPING, PARTICLE_R],
                      device="cuda:0")
        else:
            wp.launch(physics_step, dim=N_ICE,
                      inputs=[self.ice_pos, self.ice_vel,
                              float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                              half_l, half_w, half_h,
                              float(blade_dir), float(self.lean),
                              self.blade_force, self.pen_out,
                              N_ICE, DT, STIFFNESS, DAMPING, PARTICLE_R],
                      device="cuda:0")

        wp.launch(pen_reduce, dim=N_ICE,
                  inputs=[self.pen_out, self.pen_stats, N_ICE],
                  device="cuda:0")
        wp.synchronize()

    # ── settle blade Z ────────────────────────────────────────────

    def settle_blade_quick(self, steps=50):
        """Position blade so its lowest point touches ice surface (z=0).
        Physics engine will naturally calculate penetration from weight."""
        
        # Compute lowest point offset from blade center (in meters)
        # Returns negative value = lowest point is below blade center
        pitch_rad = self.pitch * 5.0 * math.pi / 180.0
        lowest_offset = 0.0
        if self.blade_geom is not None:
            lowest_offset, lowest_x = self.blade_geom.get_lowest_point_offset(
                self.lean, pitch_rad)
        else:
            # Fallback: 0.75 scaled = 15mm = blade center to edge
            lowest_offset = -0.015 * math.cos(self.lean)
        
        # Position blade center so lowest point touches ice surface (ICE_H)
        # lowest_offset is negative, so we subtract it to raise the center
        lowest_offset_scaled = lowest_offset * SCALE
        target_z = ICE_H - lowest_offset_scaled
        
        print(f"[settle] lowest_offset={lowest_offset*1000:.3f}mm, target_Z={target_z:.4f}", flush=True)

        self.pos[2] = target_z
        self.vel   = np.array([0.0, 0.0, 0.0])
        self.vel_z = 0.0

        wp.launch(init_ice, dim=N_ICE,
                  inputs=[self.ice_pos, self.ice_vel,
                          N_ICE, ICE_L, ICE_W, ICE_H,
                          self.recenter_seed + self.frame],
                  device="cuda:0")
        wp.synchronize()

        best_pen, best_cnt, best_sum = 0.0, 0, 0.0
        for _ in range(steps):
            self._step_particles()
            ps  = self.pen_stats.numpy()
            cnt = int(ps[2])
            if cnt > best_cnt:
                best_pen = float(ps[0])
                best_sum = float(ps[1])
                best_cnt = cnt

        print(f"  [settle] best: particle_pen={best_pen*1000/SCALE:.3f}mm cnt={best_cnt}", flush=True)

        edge_z     = self.pos[2] - 0.75 * math.cos(self.lean)
        geo_pen_mm = max(0.0, (ICE_H - edge_z) / SCALE * 1000.0)

        self.pen_max_mm        = geo_pen_mm
        self.pen_avg_mm        = geo_pen_mm * 0.6
        self.pen_contact_count = best_cnt
        particle_r_real_m      = PARTICLE_R / SCALE
        particle_area_mm2      = math.pi * (particle_r_real_m * 1000) ** 2
        self.pen_contact_area_mm2 = best_cnt * particle_area_mm2

        if self.blade_geom is not None and self.pen_max_mm > 0.001:
            lean_deg  = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0
            clen, cwid, carea = self.blade_geom.query(lean_deg, pitch_deg, self.pen_max_mm)
            self.contact_length_mm = clen * 1000
            self.contact_width_mm  = cwid * 1000
            self.contact_area_mm2  = carea * 1e6

        self.vel_z = 0.0
        self.vel   = np.array([0.0, 0.0, 0.0])
        self.blade_force.zero_()
        self.force_accum_along = 0.0
        self.force_accum_perp  = 0.0

        print(f"[settle] DONE Z={self.pos[2]:.4f}, GPU_pen={self.pen_max_mm:.3f}mm", flush=True)

    # ── main simulation step ──────────────────────────────────────

    def step(self):
        self.frame += 1

        # Push force
        fx, fy = 0.0, 0.0
        if self.push_frames > 0:
            F = 5000.0 * self.force_mult
            blade_dir = self.yaw + self.alpha
            cos_b = math.cos(blade_dir)
            sin_b = math.sin(blade_dir)
            fx = self.push_fx * cos_b - self.push_fy * sin_b
            fy = self.push_fx * sin_b + self.push_fy * cos_b
            fx *= F
            fy *= F
            self.push_frames -= 1

        self._step_particles()

        # Read reaction force
        bf = self.blade_force.numpy()[0]
        reaction_fx = float(bf[0])
        reaction_fy = float(bf[1])
        reaction_fz = float(bf[2])
        self.reaction_fz_real = reaction_fz / SCALE

        # Penetration (geometric)
        edge_z     = self.pos[2] - 0.75 * math.cos(self.lean)
        geo_pen_mm = max(0.0, (ICE_H - edge_z) / SCALE * 1000.0)
        d = self.force_decay
        self.pen_max_mm = d * self.pen_max_mm + (1 - d) * geo_pen_mm
        self.pen_avg_mm = self.pen_max_mm * 0.6

        ps = self.pen_stats.numpy()
        pen_count = int(ps[2])
        self.pen_contact_count = pen_count
        particle_r_real_m = PARTICLE_R / SCALE
        particle_area_mm2 = math.pi * (particle_r_real_m * 1000) ** 2
        self.pen_contact_area_mm2 = (d * self.pen_contact_area_mm2
                                     + (1 - d) * (pen_count * particle_area_mm2))

        # Contact geometry lookup
        if self.blade_geom is not None and self.pen_max_mm > 0.01:
            lean_deg  = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0
            clen, cwid, carea = self.blade_geom.query(lean_deg, pitch_deg, self.pen_max_mm)
            self.contact_length_mm = d * self.contact_length_mm + (1 - d) * (clen * 1000)
            self.contact_width_mm  = d * self.contact_width_mm  + (1 - d) * (cwid * 1000)
            self.contact_area_mm2  = d * self.contact_area_mm2  + (1 - d) * (carea * 1e6)

        # Analytical penetration
        if self.blade_geom is not None:
            F_normal = self.blade_mass * 9.81 * math.cos(self.lean)
            H_pa     = self.ice_hardness_mpa * 1e6
            lean_deg  = abs(math.degrees(self.lean))
            pitch_deg = self.pitch * 5.0
            raw = self.blade_geom.solve_depth(F_normal, H_pa, lean_deg, pitch_deg)
            self.pen_analytical_mm = d * self.pen_analytical_mm + (1 - d) * raw

        # Force accumulation (along / perp to blade)
        blade_dir = self.yaw + self.alpha
        cos_b = math.cos(blade_dir)
        sin_b = math.sin(blade_dir)
        f_a = abs(reaction_fx * cos_b + reaction_fy * sin_b)
        f_p = abs(-reaction_fx * sin_b + reaction_fy * cos_b)
        self.force_accum_along = d * self.force_accum_along + (1 - d) * f_a
        self.force_accum_perp  = d * self.force_accum_perp  + (1 - d) * f_p

        # ── velocity update ───────────────────────────────────────
        speed_before = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
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

        # Energy conservation cap
        speed_after = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if self.push_frames > 0:
            F_push = 5000.0 * self.force_mult
            elapsed = 200 - self.push_frames
            theoretical_v = F_push * elapsed * DT / self.blade_mass * 1.1
            theoretical_v = max(theoretical_v, speed_before)
            if speed_after > theoretical_v and theoretical_v > 0.01:
                scale_f = theoretical_v / speed_after
                self.vel[0] *= scale_f
                self.vel[1] *= scale_f
            self.peak_push_speed = max(self.peak_push_speed, theoretical_v)
        elif self.peak_push_speed > 0.01 and speed_after > self.peak_push_speed:
            scale_f = self.peak_push_speed / speed_after
            self.vel[0] *= scale_f
            self.vel[1] *= scale_f

        if self.push_frames == 195:
            speed_now = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
            print(f"[step] push active: fx={fx:.1f} fy={fy:.1f} "
                  f"rx={reaction_fx:.1f} ry={reaction_fy:.1f} "
                  f"ax={ax:.1f} vel=({self.vel[0]:.6f},{self.vel[1]:.6f}) "
                  f"spd_real={speed_now/SCALE:.6f}")

        # Speed cap (10 m/s real)
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        MAX_SPEED = 10.0 * SCALE
        if speed > MAX_SPEED:
            scale_f = MAX_SPEED / speed
            self.vel[0] *= scale_f
            self.vel[1] *= scale_f
            speed = MAX_SPEED

        # Air drag
        if speed > 0.001:
            self.vel[0] *= 0.99999
            self.vel[1] *= 0.99999
        else:
            self.vel[0] = 0.0
            self.vel[1] = 0.0

        # ── arc turning from lean angle ───────────────────────────
        speed_real = speed / SCALE
        self.Fy = self.blade_mass * G

        if abs(self.lean) > 0.01 and speed > 0.5:
            cos_y = math.cos(self.yaw)
            sin_y = math.sin(self.yaw)
            v_along = self.vel[0] * cos_y + self.vel[1] * sin_y
            v_along_real = v_along / SCALE

            if abs(v_along) > 0.5:
                idx = max(0, min(3, int((self.pitch + 1) / 2 * 3.99)))
                _, R_real = ROCKER_ZONES[idx]
                lean_sign   = 1.0 if self.lean > 0 else -1.0
                lean_factor = min(1.0, abs(self.lean) / (math.pi / 4))

                R_eff_real   = R_real / lean_factor if lean_factor > 0.001 else 1e6
                R_eff_scaled = R_eff_real * SCALE

                self.ac = (v_along_real ** 2) / R_eff_real if R_eff_real > 0.01 else 0.0
                self.Fx = self.blade_mass * self.ac

                if R_eff_real > 0.01 and speed_real > 0.01:
                    self.theta_balance = math.atan2(speed_real ** 2, R_eff_real * G)
                else:
                    self.theta_balance = 0.0

                omega = (v_along / R_eff_scaled) * lean_factor * lean_sign
                self.yaw += omega * DT

                new_cos = math.cos(self.yaw)
                new_sin = math.sin(self.yaw)
                v_perp  = -self.vel[0] * sin_y + self.vel[1] * cos_y
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

        # Centre of mass
        theta = abs(self.lean)
        self.G_pos = [
            self.pos[0] / SCALE,
            self.pos[1] / SCALE,
            self.L * math.cos(theta),
        ]

        # Position update
        self.pos[0] += self.vel[0] * DT
        self.pos[1] += self.vel[1] * DT

        return self.get_state()

    # ── serialise state for WebSocket ─────────────────────────────

    def get_state(self):
        speed_scaled = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        speed_real   = speed_scaled / SCALE

        blade_dir = self.yaw + self.alpha
        bx, by = math.cos(blade_dir), math.sin(blade_dir)
        va = (self.vel[0] * bx + self.vel[1] * by) / SCALE
        ppx, ppy = -by, bx
        vp = (self.vel[0] * ppx + self.vel[1] * ppy) / SCALE

        idx = max(0, min(3, int((self.pitch + 1) / 2 * 3.99)))
        zone_name, R = ROCKER_ZONES[idx]

        f_along_disp = self.force_accum_along / SCALE
        f_perp_disp  = self.force_accum_perp  / SCALE
        pos_real = [self.pos[0] / SCALE, self.pos[1] / SCALE, self.pos[2] / SCALE]
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
            'pen': round(self.pen_max_mm, 4),
            'pen_max_mm': round(self.pen_max_mm, 4),
            'pen_avg_mm': round(self.pen_avg_mm, 4),
            'pen_contact_count': self.pen_contact_count,
            'pen_contact_area_mm2': round(self.pen_contact_area_mm2, 2),
            'Lc': round(self.contact_length_mm / 1000, 4),
            'contact_length_mm': round(self.contact_length_mm, 2),
            'contact_width_mm': round(self.contact_width_mm, 3),
            'contact_area_geom_mm2': round(self.contact_area_mm2, 2),
            'pen_analytical_mm': round(self.pen_analytical_mm, 4),
            'ice_hardness_mpa': round(self.ice_hardness_mpa, 2),
            'blade_reaction_z': round(self.reaction_fz_real, 1),
            'F_normal': round(self.blade_mass * G * math.cos(self.lean), 1),
            'R': round(R, 4),
            'zone': idx,
            'zone_name': zone_name,
            'contact_z': 0.0,
            'L': round(self.L, 4),
            'G_pos': [round(self.G_pos[0], 6), round(self.G_pos[1], 6), round(self.G_pos[2], 6)],
            'Fx': round(self.Fx, 2),
            'Fy': round(self.Fy, 2),
            'Fg': round(self.blade_mass * G, 2),
            'mass': round(self.blade_mass, 1),
            'ac': round(self.ac, 4),
            'theta_balance': round(math.degrees(self.theta_balance), 2),
            'theta_actual': round(math.degrees(abs(self.lean)), 2),
            'alpha': round(math.degrees(self.alpha), 2),
            'frame': self.frame,
            'engine': 'warp-gpu-groove',
            'collision_mode': 'mesh' if self.blade_mesh is not None else 'box',
            'hollow_radius_mm': round(self.hollow_radius * 1000, 2),
            'f_along': round(f_along_disp, 1),
            'f_lateral': round(f_perp_disp, 1),
            'n_ice': N_ICE,
        }

    # ── command handler ───────────────────────────────────────────

    def handle_command(self, cmd):
        global STIFFNESS
        t = cmd.get('cmd', '') or cmd.get('type', '')

        if t == 'push':
            self.push_fx = cmd.get('fx', 0)
            self.push_fy = cmd.get('fy', 0)
            fv = cmd.get('force', None)
            if fv is not None:
                self.force_mult = fv
            self.push_frames = 200
            print(f"[cmd] PUSH fx={self.push_fx}, fy={self.push_fy}, "
                  f"force={self.force_mult}, frames={self.push_frames}")

        elif t == 'lean':
            deg = cmd.get('value', 15)
            self.lean = deg * math.pi / 180
            self.settle_blade_quick()

        elif t == 'set_L':
            self.L = max(0.3, min(1.5, cmd.get('value', 0.9)))

        elif t == 'alpha':
            deg = max(-90, min(90, cmd.get('value', 0)))
            self.alpha = deg * math.pi / 180
            print(f'[cmd] Alpha (foot opening) = {deg}\u00B0')
            self.settle_blade_quick()

        elif t == 'set_velocity':
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
            radius_m  = radius_mm / 1000.0
            if abs(radius_m - self.hollow_radius) > 1e-6 and os.path.exists(STL_PATH):
                self.hollow_radius = radius_m
                try:
                    self.blade_mesh, self.mesh_data = load_blade_mesh(
                        STL_PATH, SCALE, self.hollow_radius)
                    print(f'[cmd] Hollow radius={radius_mm:.1f}mm')
                    self.settle_blade_quick()
                except Exception as e:
                    print(f'[cmd] Hollow radius change failed: {e}')

        elif t == 'toggle_mesh':
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
            self.pos = np.array([0.0, 0.0, ICE_H + 0.75 * math.cos(self.lean)])
            self.vel = np.array([0.0, 0.0, 0.0])
            self.yaw   = 0.0
            self.lean   = 15.0 * math.pi / 180
            self.alpha  = 0.0
            self.pitch  = 0.0
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
            self.force_accum_perp  = 0.0
            wp.launch(init_ice, dim=N_ICE,
                      inputs=[self.ice_pos, self.ice_vel,
                              N_ICE, ICE_L, ICE_W, ICE_H, 42],
                      device="cuda:0")
            wp.synchronize()
            self.settle_blade_quick()
            print("[cmd] RESET complete")

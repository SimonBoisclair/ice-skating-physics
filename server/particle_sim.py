"""
Particles-only pool simulation for the GPU CAD visualizer.
"""
import os
import math
import numpy as np
import warp as wp

from .config import (
    SCALE, ICE_L, ICE_W, ICE_H, N_ICE,
    CUBE_CONTACT_DAMPING, CUBE_CONTACT_FRICTION, CUBE_CONTACT_STIFFNESS,
    CUBE_DROP_GAP, CUBE_MASS, CUBE_SIZE,
    DT, G, PARTICLE_R, STIFFNESS_BASE, DAMPING, STL_PATH,
)
from .kernels import init_ice_lattice, physics_step_particles_only


class ParticlePoolSimulation:
    def __init__(self):
        self.ice_pos = wp.zeros(N_ICE, dtype=wp.vec3, device="cuda:0")
        self.ice_vel = wp.zeros(N_ICE, dtype=wp.vec3, device="cuda:0")
        self.pen_out = wp.zeros(N_ICE, dtype=float, device="cuda:0")
        self.pen_stats = wp.zeros(3, dtype=float, device="cuda:0")

        self.grain_nx = 600
        self.grain_ny = 40
        self.grain_grid = wp.HashGrid(256, 64, 64, device="cuda:0")
        self.grain_grid_cell_size = PARTICLE_R * 2.5

        self.frame = 0
        self.recenter_seed = 1000
        self.physics_paused = True
        self.particles_only_mode = True
        self.ice_hardness_mpa = 4.75
        self.mesh_data = self._load_blade_cad_data()
        self.cube_size = CUBE_SIZE
        self.cube_mass = CUBE_MASS
        self.substeps = 8
        self.cube_pos = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        self.cube_vel = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        self.cube_force = wp.zeros(3, dtype=float, device="cuda:0")
        self.cube_mesh_verts, self.cube_mesh_faces = self._build_cube_collider_mesh()

        self.reset_particles()
        print("[physics] Particles-only pool simulation ready.", flush=True)

    def _build_cube_collider_mesh(self):
        h = CUBE_SIZE * 0.5
        verts = np.array([
            (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
            (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
        ], dtype=np.float32)

        # Rotate 45 deg around X, then 45 deg around Y -> corner points down
        angle = math.pi / 4.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Rx(45)
        rotated = verts.copy()
        y_new = rotated[:, 1] * cos_a - rotated[:, 2] * sin_a
        z_new = rotated[:, 1] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 1] = y_new
        rotated[:, 2] = z_new
        # Ry(45)
        x_new = rotated[:, 0] * cos_a + rotated[:, 2] * sin_a
        z_new2 = -rotated[:, 0] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 0] = x_new
        rotated[:, 2] = z_new2
        verts = rotated
        faces = np.array([
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ], dtype=np.int32)
        return (
            wp.array(verts, dtype=wp.vec3, device="cuda:0"),
            wp.array(faces, dtype=int, device="cuda:0"),
        )

    def _load_blade_cad_data(self):
        if not os.path.exists(STL_PATH):
            return None
        try:
            from stl import mesh as stl_mesh
            m = stl_mesh.Mesh.from_file(STL_PATH)
            verts = m.vectors.reshape(-1, 3).astype(np.float32)
            faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
            verts[:, 0] *= SCALE
            verts[:, 1] *= SCALE
            verts[:, 2] *= SCALE
            return {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "n_verts": int(len(verts)),
                "n_faces": int(len(faces)),
            }
        except Exception as e:
            print(f"[physics] Blade CAD data load skipped: {e}", flush=True)
            return None

    def reset_particles(self):
        self.recenter_seed += 1
        wp.launch(
            init_ice_lattice,
            dim=N_ICE,
            inputs=[
                self.ice_pos,
                self.ice_vel,
                N_ICE,
                ICE_L,
                ICE_W,
                ICE_H,
                self.grain_nx,
                self.grain_ny,
                self.recenter_seed,
            ],
            device="cuda:0",
        )
        self.pen_out.zero_()
        self.pen_stats.zero_()
        self.reset_cube()
        wp.synchronize()

    def reset_cube(self):
        self.cube_pos.assign(np.array([(0.0, 0.0, ICE_H + CUBE_DROP_GAP + CUBE_SIZE * 0.5)], dtype=np.float32))
        self.cube_vel.assign(np.array([(0.0, 0.0, 0.0)], dtype=np.float32))
        self.cube_force.zero_()

    def step(self):
        if self.physics_paused:
            return self.get_state()

        self.frame += 1
        sub_dt = DT / self.substeps
        for _ in range(self.substeps):
            self.pen_out.zero_()
            self.pen_stats.zero_()
            self.cube_force.zero_()
            self.grain_grid.build(self.ice_pos, self.grain_grid_cell_size)
            wp.launch(
                physics_step_particles_only,
                dim=N_ICE,
                inputs=[
                    self.grain_grid.id,
                    self.ice_pos,
                    self.ice_vel,
                    self.pen_out,
                    self.cube_pos,
                    self.cube_vel,
                    self.cube_force,
                    self.cube_mesh_verts,
                    self.cube_mesh_faces,
                    len(self.cube_mesh_faces.numpy()) // 3,
                    N_ICE,
                    sub_dt,
                    STIFFNESS_BASE * (self.ice_hardness_mpa / 7.0),
                    DAMPING,
                    PARTICLE_R,
                    ICE_L,
                    ICE_W,
                    ICE_H,
                    self.cube_size,
                    CUBE_CONTACT_STIFFNESS,
                    CUBE_CONTACT_DAMPING,
                    CUBE_CONTACT_FRICTION,
                ],
                device="cuda:0",
            )
            wp.synchronize()

            cube_pos = self.cube_pos.numpy()[0]
            cube_vel = self.cube_vel.numpy()[0]
            cube_force = self.cube_force.numpy()
            cube_force[2] -= self.cube_mass * G * SCALE
            cube_vel = cube_vel + (cube_force / self.cube_mass) * sub_dt
            cube_pos = cube_pos + cube_vel * sub_dt
            self.cube_pos.assign(np.array([cube_pos], dtype=np.float32))
            self.cube_vel.assign(np.array([cube_vel], dtype=np.float32))
        return self.get_state()

    def get_state(self):
        cube_pos = self.cube_pos.numpy()[0]
        cube_vel = self.cube_vel.numpy()[0]
        cube_force = self.cube_force.numpy()
        return {
            "type": "state",
            "speed": 0.0,
            "f_lateral": 0.0,
            "f_along": 0.0,
            "n_ice": N_ICE,
            "physics_paused": self.physics_paused,
            "particles_only_mode": True,
            "ice_hardness_mpa": round(self.ice_hardness_mpa, 2),
            "cube_pos": cube_pos.tolist(),
            "cube_vel": cube_vel.tolist(),
            "cube_force": cube_force.tolist(),
        }

    def handle_command(self, cmd):
        t = cmd.get("cmd", "") or cmd.get("type", "")
        print(f"[handle_command] Received: {cmd}, type={t}", flush=True)

        if t in ("reset", "reset_particles_only"):
            self.physics_paused = True
            self.reset_particles()
            print("[cmd] PARTICLES RESET complete", flush=True)
        elif t == "start_particles_only":
            self.physics_paused = False
            print("[cmd] Starting particles-only pool simulation", flush=True)
        elif t == "pause":
            self.physics_paused = True
            print("[cmd] Physics paused", flush=True)
        elif t == "ice":
            hardness = max(1, min(20, cmd.get("value", 7)))
            self.ice_hardness_mpa = hardness
            print(f"[cmd] Ice hardness={hardness} MPa", flush=True)

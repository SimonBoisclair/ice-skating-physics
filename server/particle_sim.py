"""
Particles-only pool simulation for the GPU CAD visualizer.
"""
import math
import numpy as np
import warp as wp

from .config import (
    SCALE, ICE_L, ICE_W, ICE_H, N_ICE,
    CUBE_CONTACT_DAMPING, CUBE_CONTACT_FRICTION, CUBE_CONTACT_STIFFNESS,
    CUBE_DROP_GAP, CUBE_MASS, CUBE_SIZE,
    DT, G, PARTICLE_R, STIFFNESS_BASE, CONTACT_DAMPING,
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
        self.cube_size = CUBE_SIZE
        self.cube_mass = CUBE_MASS
        self.substeps = 8
        self.cube_pos = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        self.cube_vel = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        self.cube_force = wp.zeros(3, dtype=float, device="cuda:0")
        self.cube_mesh_verts, self.cube_mesh_faces = self._build_cube_collider_mesh()
        self.cube_corner_depth = float(abs(min(v[2] for v in self.cube_mesh_verts.numpy())))

        self.reset_particles()
        print("[physics] Simulation ready.", flush=True)

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
        rotated = verts.copy()
        y_new = rotated[:, 1] * cos_a - rotated[:, 2] * sin_a
        z_new = rotated[:, 1] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 1] = y_new
        rotated[:, 2] = z_new
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

    def reset_particles(self):
        self.recenter_seed += 1
        wp.launch(
            init_ice_lattice,
            dim=N_ICE,
            inputs=[
                self.ice_pos, self.ice_vel, N_ICE,
                ICE_L, ICE_W, ICE_H,
                self.grain_nx, self.grain_ny, self.recenter_seed,
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
            return

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
                    self.ice_pos, self.ice_vel, self.pen_out,
                    self.cube_pos, self.cube_vel, self.cube_force,
                    self.cube_mesh_verts, self.cube_mesh_faces,
                    len(self.cube_mesh_faces.numpy()) // 3,
                    N_ICE, sub_dt,
                    STIFFNESS_BASE, PARTICLE_R,
                    ICE_L, ICE_W, ICE_H,
                    self.cube_size,
                    CUBE_CONTACT_STIFFNESS, CUBE_CONTACT_DAMPING, CUBE_CONTACT_FRICTION,
                    CONTACT_DAMPING,
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
            if cube_pos[2] < self.cube_corner_depth:
                cube_pos[2] = self.cube_corner_depth
                cube_vel[2] = 0.0
            self.cube_pos.assign(np.array([cube_pos], dtype=np.float32))
            self.cube_vel.assign(np.array([cube_vel], dtype=np.float32))

    def handle_command(self, cmd):
        t = cmd.get("cmd", "") or cmd.get("type", "")
        if t in ("reset", "reset_particles_only"):
            self.physics_paused = True
            self.reset_particles()
        elif t == "start_particles_only":
            self.physics_paused = False
        elif t == "pause":
            self.physics_paused = True

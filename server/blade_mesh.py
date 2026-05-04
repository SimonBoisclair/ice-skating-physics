"""
Load a blade STL file, transform it to simulation coordinates, and
create a Warp GPU mesh for collision queries.
"""
import math
import numpy as np
import warp as wp

from .config import BLADE_H_REAL, HOLLOW_RADIUS_DEFAULT


def load_blade_mesh(stl_path, scale, hollow_radius=HOLLOW_RADIUS_DEFAULT):
    """Load blade STL, transform to simulation local frame, create wp.Mesh.

    STL coordinate system:
      X = blade length, Y = height (0 = bottom edge), Z = thickness

    Simulation blade-local frame:
      X = along blade, Y = across blade (thickness), Z = height
    """
    import trimesh

    mesh = trimesh.load(stl_path)
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()

    print(f"[mesh] Loaded STL: {verts.shape[0]} vertices, {faces.shape[0]} triangles")
    print(f"[mesh] STL bounds: X=[{verts[:,0].min():.4f}, {verts[:,0].max():.4f}], "
          f"Y=[{verts[:,1].min():.4f}, {verts[:,1].max():.4f}], "
          f"Z=[{verts[:,2].min():.4f}, {verts[:,2].max():.4f}]")

    # Adjust hollow radius by modifying bottom vertices
    if abs(hollow_radius - HOLLOW_RADIUS_DEFAULT) > 1e-6:
        R_new = hollow_radius
        half_t = 0.0015
        bottom_mask = verts[:, 1] < 0.003
        for i in range(len(verts)):
            if bottom_mask[i]:
                z = verts[i, 2]
                if abs(z) < half_t:
                    y_new = math.sqrt(R_new**2 - z**2) - math.sqrt(R_new**2 - half_t**2)
                    verts[i, 1] = max(0.0, y_new)
        print(f"[mesh] Adjusted hollow radius to {hollow_radius*1000:.2f}mm")

    # Transform: STL (X,Y,Z) → blade-local (X,Y,Z)
    blade_center_y = BLADE_H_REAL / 2.0

    transformed = np.zeros_like(verts)
    transformed[:, 0] = verts[:, 0] * scale
    transformed[:, 1] = verts[:, 2] * scale
    transformed[:, 2] = (verts[:, 1] - blade_center_y) * scale

    print(f"[mesh] Transformed bounds: X=[{transformed[:,0].min():.2f}, {transformed[:,0].max():.2f}], "
          f"Y=[{transformed[:,1].min():.2f}, {transformed[:,1].max():.2f}], "
          f"Z=[{transformed[:,2].min():.2f}, {transformed[:,2].max():.2f}]")

    # Fix normals
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    # Create Warp mesh on GPU
    wp_points  = wp.array(transformed.astype(np.float32), dtype=wp.vec3, device="cuda:0")
    wp_indices = wp.array(faces.flatten().astype(np.int32), dtype=wp.int32, device="cuda:0")

    wp_mesh = wp.Mesh(
        points=wp_points,
        indices=wp_indices,
        support_winding_number=True,
    )

    print(f"[mesh] wp.Mesh created on GPU (id={wp_mesh.id}, support_winding_number=True)")

    mesh_data = {
        'vertices': transformed.tolist(),
        'faces': faces.tolist(),
        'n_verts': int(verts.shape[0]),
        'n_faces': int(faces.shape[0]),
        'hollow_radius': hollow_radius,
    }

    return wp_mesh, mesh_data

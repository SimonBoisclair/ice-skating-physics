"""
Constants and simulation parameters.

SCALE must match the frontend (sandbox-ui/src/constants.js).
All "scaled" quantities = real_meters * SCALE.
"""
import os

# ─── Mesh / CAD ───
HOLLOW_RADIUS_DEFAULT = 0.015875  # 5/8" = 15.875 mm in metres
STL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "blade-holder-cad.stl")
USE_MESH_COLLISION = True

# ─── Scale factor ───
SCALE = 50  # 1 m real → 50 sim-units

# ─── Blade dimensions (real metres) ───
BLADE_LEN_REAL = 0.280
BLADE_W_REAL   = 0.003
BLADE_H_REAL   = 0.030

# Scaled blade dimensions
BLADE_LEN = BLADE_LEN_REAL * SCALE  # 14.0
BLADE_W   = BLADE_W_REAL   * SCALE  # 0.15
BLADE_H   = BLADE_H_REAL   * SCALE  # 1.5

# ─── Ice particle field (tight box around blade contact zone) ───
ICE_L = 0.300 * SCALE   # 300 mm → 15.0 scaled
ICE_W = 0.025 * SCALE   # 25 mm  → 1.25 scaled (handles lean sweep up to 45°)
ICE_H = 0.005 * SCALE   # 5 mm   → 0.25 scaled (particle pool depth)
ICE_SHEET = 0.015 * SCALE  # 15 mm → 0.75 scaled (total ice sheet thickness)
N_ICE = 240_000

# ─── Physics parameters ───
DT          = 0.001
G           = 9.81
ICE_RHO     = 917.0
PARTICLE_R  = 0.025 * SCALE / 50.0   # ~0.5 mm real
STIFFNESS_BASE = 2e5
DAMPING     = 80.0
BLADE_MASS  = 85.0   # kg (default skater mass)

# ─── Centre of mass ───
L_COM = 0.90  # metres (P → G distance, crouched skater)

# ─── Rocker zones (Quad 1 profile) ───
ROCKER_ZONES = [
    ("Zone 1 (6')",  6  * 0.3048),
    ("Zone 2 (9')",  9  * 0.3048),
    ("Zone 3 (12')", 12 * 0.3048),
    ("Zone 4 (15')", 15 * 0.3048),
]

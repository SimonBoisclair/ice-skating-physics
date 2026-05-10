"""
Constants and simulation parameters.

All "scaled" quantities = real_meters * SCALE.
"""

# ─── Scale factor ───
SCALE = 50  # 1 m real → 50 sim-units

# ─── Particle pool dimensions ───
ICE_L = 0.250 * SCALE   # 250 mm → 12.5 scaled (matches CAD pool groove)
ICE_W = 0.025 * SCALE   # 125 mm → 6.25 scaled
ICE_H = 0.005 * SCALE   # 5 mm   → 0.25 scaled (particle pool depth)
ICE_SHEET = 0.015 * SCALE  # 15 mm → 0.75 scaled (total ice sheet thickness)
N_ICE = 240_000

# ─── Falling cube ───
CUBE_SIZE = 0.005 * SCALE
CUBE_DROP_GAP = 0.010 * SCALE    # 10 mm realistic drop height
CUBE_MASS = 0.5                   # 500 g test weight
CUBE_CONTACT_STIFFNESS = 500_000  # same as ice-ice contacts
CUBE_CONTACT_DAMPING = 50_000     # same as ice-ice contacts
CUBE_CONTACT_FRICTION = 0.5

# ─── DEM physics parameters ───
DT          = 0.001
G           = 9.81
ICE_RHO     = 917.0
PARTICLE_R  = 0.025 * SCALE / 50.0   # ~0.5 mm real
STIFFNESS_BASE = 500_000              # normal contact spring stiffness
CONTACT_DAMPING = 50_000              # dashpot coefficient (overdamped → e ≈ 0)

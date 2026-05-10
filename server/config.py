"""
Constants and simulation parameters.

All "scaled" quantities = real_meters * SCALE.
"""
import os

# ─── Mesh / CAD ───
STL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "blade-holder-cad.stl")

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
CUBE_DROP_GAP = 0.030 * SCALE  # 30mm drop height for impact velocity
CUBE_MASS = 2.0               # 2 kg test weight for visible impact
CUBE_CONTACT_STIFFNESS = 100.0    # moderate so cube penetrates to make dents
CUBE_CONTACT_DAMPING = 20.0       # high damping, no bounce
CUBE_CONTACT_FRICTION = 0.5

# ─── Physics parameters ───
DT          = 0.001
G           = 9.81
ICE_RHO     = 917.0
PARTICLE_R  = 0.025 * SCALE / 50.0   # ~0.5 mm real
STIFFNESS_BASE = 500_000          # hard ice (was 2e5)
DAMPING     = 0.96               # multiplicative velocity damping per substep
CONTACT_DAMPING     = 0.96               # multiplicative velocity damping per substep
GRID_RESTORATION_FACTOR = 0.0     # disabled -> permanent dents
RESTITUTION = 0.03                # almost no bounce
VEL_DAMP_POST = 0.92              # extra post-collision velocity damping

# Ice Skating Physics Engine — Technical Document

## Overview

The physics engine has **two layers**:

1. **Static Sandbox** (`sandbox.html`) — analytical equations, no time evolution. Computes all physics from input parameters instantly. This is what the current 3D sandbox uses.
2. **GPU Particle Simulation** (`warp_server.py`) — NVIDIA Warp-based particle physics running on GPU. Ice is 18,000 particles; the blade physically pushes through them. Friction emerges from geometry, not coefficients. Used by the dashboard/test runner.

This document covers both, with emphasis on **how penetration and resistance are currently calculated**.

---

## 1. Ice Penetration

### 1.1 Static Sandbox (sandbox.html)

**Input parameters that affect penetration:**
- `F_normal` — normal force into ice (≈ m·g = 833.9N for 85kg skater)
- `ice_hardness` (H) — computed from ice temperature
- `contact_length` — how much blade touches ice (depends on rocker zone, force, hardness)
- `lean_angle` — concentrates force on one edge (increases depth)

**Step 1: Ice Hardness from Temperature**

```
H = 0.65 × |T| + 1.5 MPa     (Barnes & Tabor 1966, Poirier et al. 2011)
clamped to [1, 20] MPa

Examples:
  -1°C (warm/soft):  H = 2.15 MPa
  -5°C (typical rink): H = 4.75 MPa
  -20°C (outdoor cold): H = 14.5 MPa
```

**Step 2: Contact Length**

Base contact length is 40mm at center rocker zone, scaled by force and hardness:

```
contact_length = 40mm × √(F_normal / 833.9N) × √(5 MPa / H)

Adjustments:
  - Pitch near tips (|pitch| > 0.6): ×0.5 (shorter contact)
  - Pitch in transition zones (|pitch| > 0.2): ×0.75
  - Clamped to [5mm, 224mm] (max 80% of blade length)
```

This is a simplified Hertz contact approximation (cylinder on flat surface):
`a = √(4FR / πEL)`, where R = rocker radius, F = force, E = elastic modulus, L = blade length.

**Step 3: Penetration Depth**

```
depth = F_normal / (H_pa × contact_length × 50)

where:
  H_pa = H in Pascals (H_mpa × 1e6)
  50 = empirical width scaling factor

Then multiply by lean factor:
  lean_factor = 1.0 + sin(lean_angle) × 0.5

Clamped to [0.1mm, 5mm]
```

**Typical result:** At -5°C, 85kg, 15° lean → depth ≈ 0.11mm, contact length ≈ 41mm

**Step 4: Polygon Width**

```
width = BLADE_T × 0.5   (if lean > 5°, single edge contact → 1.5mm)
width = BLADE_T          (if lean ≤ 5°, both edges → 3mm)
```

**Step 5: Contact Edge**

```
lean < 2°  → "both" edges
lean ≥ 2°  → "inside" or "outside" depending on G.y relative to P.y
```

### 1.2 GPU Particle Simulation (warp_server.py)

In the particle simulation, penetration is **emergent** — not calculated by a formula. Instead:

- 18,000 ice particles fill a volume around the blade
- The blade is a rigid box (280mm × 3mm × 30mm, scaled 50×)
- Each particle checks if it overlaps the blade box (in blade-local coordinates)
- Overlapping particles get pushed out with spring forces:
  - `force = stiffness × penetration_depth`
  - Stiffness is set by ice hardness (default 2e5)
  - Penetration capped at 3× particle radius to prevent explosions

The blade literally carves a groove in the particle field. The groove shape = the penetration polygon.

---

## 2. Resistance (Friction)

### 2.1 Static Sandbox — Analytical Resistance

Velocity is decomposed into **along-blade** and **across-blade** components:

```
blade_direction = [cos(α), sin(α)]     (α = foot opening angle)

v_along  = Gvx × cos(α) + Gvy × sin(α)    (velocity along blade)
v_across = -Gvx × sin(α) + Gvy × cos(α)   (velocity across blade)
```

**Along-groove resistance (R_along):** Very low — this is why skating works.

```
R_along = μ × F_normal × sign(v_along)

where μ = 0.005 (along-groove friction coefficient)

Example: 833.9N × 0.005 = 4.2N
```

This represents the thin film of meltwater under the blade reducing friction. The blade glides freely along its groove direction.

**Across-groove resistance (R_across):** Very high — groove walls block lateral motion.

```
R_across = H_pa × groove_area × 0.001 × sign(v_across)

where:
  groove_area = contact_length × penetration_depth
  H_pa = ice hardness in Pascals
  0.001 = empirical scaling factor

Example at α=30°:
  groove_area = 0.041m × 0.00011m = 4.5e-6 m²
  R_across = 4.8e6 Pa × 4.5e-6 m² × 0.001 = 21.6N
```

This is the key asymmetry: resistance along the blade is ~100-1000× lower than across the blade. The groove walls physically prevent lateral sliding.

**Normal resistance (R_normal):** Equal and opposite to F_normal — supports the skater's weight.

```
R_normal = F_normal = m × g = 833.9N
```

### 2.2 GPU Particle Simulation — Emergent Anisotropic Friction

In the particle simulation, friction emerges from **anisotropic collision stiffness**:

```
Blade local frame stiffness:
  k_along   = stiffness × 0.002   (500× weaker along blade axis)
  k_lateral = stiffness × 1.0     (full stiffness for groove walls)
  k_vertical = stiffness × 1.0    (full stiffness for vertical)
```

**Why anisotropic?** The blade's cross-section is:
- End face (along-blade direction): 3mm × 30mm = 90mm² — tiny area, cuts through easily
- Side face (across-blade direction): 280mm × 30mm = 8,400mm² — huge area, walls resist

The stiffness ratio (0.002 = 1/500) models this area ratio. When the blade moves:
- **Along its groove:** Particles at the thin end face get weak pushback → blade slides easily
- **Across its groove:** Particles at the wide side face get strong pushback → blade is blocked

This naturally produces the along/across friction asymmetry without any friction coefficient.

**Collision detection** works in blade-local coordinates:

```
1. Transform particle position to blade frame (undo yaw, undo lean)
2. Check overlap with blade box (half-extents + particle radius)
3. Find minimum penetration axis (X=along, Y=lateral, Z=vertical)
4. Apply spring force along that axis with appropriate stiffness
5. Transform force back to world frame
6. Apply Newton's 3rd law: reaction force on blade
```

---

## 3. Other Key Physics

### 3.1 Turn Radius

```
R_turn = R_rocker / lean_factor

where:
  R_rocker = rocker zone radius (Zone 1: 1.83m, Zone 2: 2.74m, Zone 3: 3.66m, Zone 4: 4.57m)
  lean_factor = min(1.0, lean_angle / 45°)
```

More lean → tighter turn. The rocker zone sets the base radius.

### 3.2 Balance Equation (from the article)

```
θ_balance = arctan(v² / (R × g))

This is the lean angle REQUIRED for equilibrium at speed v in a turn of radius R.
If actual lean > θ_balance → OVER-LEAN (falling into turn)
If actual lean < θ_balance → UNDER-LEAN (flying outward)
```

### 3.3 Force Decomposition

```
F_gravity = m × g                          (always downward, 833.9N for 85kg)
F_body = m × g / cos(θ)                    (force along body axis G→P)
F_normal = F_gravity                        (vertical component at P)
F_tangential = F_gravity × tan(θ)           (horizontal component at P)
```

### 3.4 Center of Mass (G)

G is a free point in space defined by (Gx, Gy, Gz). It is NOT rigidly linked to P.

```
L = distance(G, P)              (computed, not fixed)
θ = arctan(horizontal_dist / Gz) (lean from vertical, computed from G position)
```

The skater controls G's position through body movement. L and θ are consequences of where G is relative to P.

### 3.5 Energy Conservation (GPU simulation only)

The particle simulation has an energy conservation guard:

```
During push: speed capped at theoretical_max = F × elapsed_frames × dt / mass × 1.1
After push: speed can only decrease (groove can't inject energy)
```

This prevents penalty collision artifacts from making the blade go faster than physically possible.

---

## 4. Constants

| Constant | Value | Description |
|----------|-------|-------------|
| BLADE_LEN | 280mm | Blade length |
| BLADE_T | 3mm | Blade thickness |
| BLADE_H | 30mm | Blade height |
| HOLLOW_R | 15.875mm | Hollow grind radius (5/8") |
| MU_ALONG | 0.005 | Along-groove friction coefficient |
| G_ACC | 9.81 m/s² | Gravitational acceleration |
| STIFFNESS | 2e5 | Base particle collision stiffness |
| DAMPING | 80.0 | Particle velocity damping |
| N_ICE | 18,000 | Number of ice particles (GPU sim) |
| SCALE | 50× | GPU sim scale factor |
| DT | 0.001s | GPU sim timestep |

---

## 5. What's Simplified / Known Limitations

1. **Penetration depth formula is empirical** — uses a scaling factor of 50 rather than deriving from Hertz contact theory rigorously. Real blade-ice contact involves plastic deformation, not elastic.

2. **Contact length is approximate** — real contact depends on the full rocker profile curve, not just the active zone radius. We use a simplified sqrt(F/H) scaling.

3. **No meltwater layer dynamics** — real skating friction depends on a thin layer of meltwater under the blade. We use a fixed μ=0.005 rather than modeling the pressure-melting and refreezing cycle.

4. **Hollow grind not modeled in penetration** — the 5/8" hollow creates two edges separated by a concave surface. The penetration polygon should technically be two thin lines (one per edge), not a rectangular cross-section. Currently we treat it as a single contact zone.

5. **No time evolution in the sandbox** — it's a static frame. The GPU particle simulation has time stepping but isn't connected to the sandbox yet.

6. **Across-groove resistance scaling is empirical** — the 0.001 factor was tuned to give reasonable forces. A more rigorous model would use the groove wall shear area and ice shear strength.

7. **Turn radius is simplified** — real turn radius depends on the full rocker profile interaction with ice at the current lean angle, not just R_rocker / lean_factor.

---

## 6. File Locations

- **Static sandbox**: `/workspace/sandbox.html` (on RunPod) / `/home/ubuntu/genesis-gpu/sandbox.html`
- **GPU physics server**: `/home/ubuntu/genesis-gpu/warp_server.py`
- **Dashboard**: `/home/ubuntu/genesis-gpu/dashboard.html`
- **Test suite**: `/home/ubuntu/genesis-gpu/test_physics.py`

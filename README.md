# Particle Pool CAD Visualizer

GPU-based particle pool visualization with CAD geometry.

The app renders:

- A premade ice/pool CAD mesh from `ice-pool-cad.stl`
- 240k granular particles in the pool
- Optional blade CAD data kept as a visual/debug asset only
- A browser `/viz` page with camera, play, pause, reset, and stream controls

There is no skate/blade physics model anymore. The active simulation is particles-only: gravity, particle-particle DEM contact, damping/friction, and pool boundary contact.

## Quick start

```bash
python3 warp_server.py
```

Open:

```text
http://localhost:8765/viz
```

The root route `/` also serves the visualization page.

## Active structure

```text
.
 warp_server.py              # Server entry point
 ice-pool-cad.stl            # Ice sheet / pool CAD
 blade-holder-cad.stl        # Blade CAD asset only
 server/
 __init__.py             # Warp GPU init    
 config.py               # Particle/CAD constants    ├─
    ├── kernels.py              # Warp particle init + DEM contact kernels
    ├── particle_sim.py         # Particles-only pool simulation
    ├── renderer_warp.py        # GPU CAD + particle MJPEG renderer
    └── web.py                  # HTTP routes, WebSocket, physics loop
```

Removed/deprecated:

- `sandbox-ui/`
- blade/skate physics state
- blade mesh collision physics
- blade geometry lookup physics
- sandbox routes (`/sandbox`)

## Main configuration

Edit `server/config.py`:

```python
SCALE = 50
ICE_L = 0.250 * SCALE
ICE_W = 0.125 * SCALE
ICE_H = 0.005 * SCALE
N_ICE = 240_000
PARTICLE_R = 0.005 * SCALE / 50.0
STIFFNESS_BASE = 2e5
DAMPING = 80.0
```

## Controls

In `/viz`:

- **Play** starts particles-only DEM simulation
- **Pause** pauses particle simulation
- **Reset** respawns particles in the pool and pauses
- Camera buttons control orbit view

## Physics currently active

The particles-only simulation uses:

- Gravity
- Warp `HashGrid` spatial neighbor search
- Particle-particle sphere contact
- Contact damping
- Tangential/friction-like resistance
- Pool bottom contact
- Pool side-wall contact

No blade or skate forces are active.

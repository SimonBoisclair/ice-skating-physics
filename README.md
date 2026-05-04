# Ice Skating Physics Simulator

Real-time GPU particle simulation of a skate blade cutting into ice.
240,000 ice particles interact with a CAD blade mesh on the GPU — friction and penetration emerge from geometry, not coefficients.

**Live demo:** https://2kppzi5i3bs1bg-8765.proxy.runpod.net/sandbox

---

## Quick Start

```bash
# 1. Start the GPU physics server (requires NVIDIA GPU + Warp)
python3 warp_server.py

# 2. Build the React frontend (first time only)
cd sandbox-ui && npm install && npm run build

# 3. Open http://localhost:8765/sandbox
```

---

## Repository Structure

```
.
├── warp_server.py              # Entry point — starts the server on port 8765
├── server/                     # Python backend (GPU physics + WebSocket)
│   ├── __init__.py             #   Warp GPU init
│   ├── config.py               #   Constants and simulation parameters
│   ├── kernels.py              #   CUDA kernels (particle physics)
│   ├── blade_mesh.py           #   STL loading and coordinate transform
│   ├── blade_geometry.py       #   Contact geometry lookup table
│   ├── physics.py              #   BladePhysics class (simulation loop)
│   └── web.py                  #   HTTP routes + WebSocket handler
├── sandbox-ui/                 # React frontend (Vite + Three.js)
│   ├── src/
│   │   ├── App.jsx             #   Root component — wires panel, scene, WebSocket
│   │   ├── constants.js        #   Shared constants (SCALE, blade dimensions)
│   │   ├── context/            #   React Context for physics state
│   │   ├── hooks/              #   useWebSocket hook
│   │   ├── components/
│   │   │   ├── Panel/          #   Input sliders and GPU readouts
│   │   │   └── Scene/          #   Three.js 3D visualisation
│   │   └── utils/              #   Blade profile and ice hardness helpers
│   └── vite.config.js
├── blade-holder-cad.stl        # Blade CAD mesh (watertight, 10k triangles)
├── blade-holder-cad.glb        # Same mesh in glTF format
├── blade-holder-cad.obj        # Same mesh in OBJ format
└── physics-engine-doc.md       # Detailed physics equations document
```

---

## Backend — `server/`

The physics server runs a continuous simulation loop on the GPU and streams
state to the browser over WebSocket.

### `config.py` — Constants

All simulation parameters live here. Key values:

| Constant | Value | Meaning |
|----------|-------|---------|
| `SCALE` | 50 | 1 real metre = 50 simulation units |
| `N_ICE` | 240,000 | Number of ice particles |
| `ICE_L/W/H` | 300 x 25 x 2 mm | Particle pool dimensions |
| `DT` | 0.001 s | Physics timestep |
| `STIFFNESS_BASE` | 200,000 | Particle collision stiffness |
| `BLADE_MASS` | 85 kg | Default skater mass |

`SCALE` must match `sandbox-ui/src/constants.js`.

### `kernels.py` — GPU Kernels

Five Warp CUDA kernels:

- **`init_ice`** — Randomly distribute particles in the ice pool volume.
- **`recenter_ice`** — Wrap particles that drift too far from the blade back to the other side (infinite ice sheet illusion).
- **`physics_step_mesh`** — Main physics kernel (mesh collision). For each particle: apply gravity, check ground plane, transform to blade-local coordinates, query signed distance to blade mesh via `wp.mesh_query_point_sign_winding_number`, apply anisotropic penalty forces, accumulate reaction force on blade.
- **`physics_step`** — Fallback kernel using box collision (when mesh unavailable).
- **`pen_reduce`** — Reduce per-particle penetration to `[max, sum, count]`.

### `blade_mesh.py` — STL Loading

`load_blade_mesh(stl_path, scale, hollow_radius)`:

1. Loads the STL with trimesh
2. Optionally adjusts hollow radius by modifying bottom vertices
3. Transforms STL coordinates → simulation coordinates
4. Creates a `wp.Mesh` on the GPU with winding number support

### `blade_geometry.py` — Contact Geometry

`BladeGeometry` pre-computes a 3D lookup table (lean x pitch x depth) at startup:

1. **Rocker profile** — Extracts the blade's bottom-edge curvature from the STL as a cubic spline. For a given pitch, tilts the ice plane and finds where the rocker curve intersects it → contact length.
2. **Hollow-grind cross-section** — Models the blade bottom as a circular arc. When leaned, computes the width of arc below the ice plane → contact width.
3. **Bisection solver** (`solve_depth`) — Finds depth `d` where `H x Lc(d) x w(d) = F_normal` (no magic multipliers).

### `physics.py` — BladePhysics

The main simulation class. Owns all state:

**Initialisation:**
- Creates GPU arrays for 240k particles
- Loads blade mesh and geometry lookup table
- Runs 200 warm-up steps to settle particles

**`step()`** — Called every frame (~250 Hz):
1. Apply push forces (if active)
2. Run particle physics (`_step_particles`)
3. Read reaction forces from GPU
4. Update penetration, contact geometry, analytical comparison
5. Update blade velocity (push + reaction forces)
6. Arc turning from lean angle (rocker zone radius)
7. Return full state dict for WebSocket broadcast

**`settle_blade_quick()`** — Called on every parameter change:
1. Compute equilibrium depth from analytical model
2. Position blade at that depth
3. Reinitialise particles and run 50 steps
4. Measure GPU contact metrics

**`handle_command(cmd)`** — WebSocket command dispatch:
- `lean`, `pitch`, `alpha` → Set angle + re-settle
- `weight` → Set mass + re-settle
- `ice` → Set hardness + adjust stiffness + re-settle
- `hollow_radius` → Reload mesh with new radius + re-settle
- `push` → Apply force for 200 frames
- `reset` → Return to defaults
- `toggle_mesh` → Switch between mesh/box collision

### `web.py` — HTTP + WebSocket

- **`physics_loop()`** — Async loop: calls `physics.step()` continuously, broadcasts state JSON to all connected clients every 4th frame.
- **`ws_handler()`** — WebSocket endpoint at `/ws`. Receives JSON commands from the browser, forwards to `physics.handle_command()`.
- **`sandbox_handler()`** — Serves the React build (`sandbox-ui/dist/index.html`) at both `/` and `/sandbox`.
- **`debug_mesh_handler()`** — Returns nearby particle positions + mesh data as JSON at `/debug_mesh`.

---

## Frontend — `sandbox-ui/`

React + Vite app with Three.js for 3D visualisation. Built with `npm run build`, served as static files by the Python server.

### Data Flow

```
User moves slider
    → PhysicsContext updates param
    → useWebSocket sends JSON command to server
    → Server runs GPU physics, broadcasts new state
    → useWebSocket receives state, updates PhysicsContext
    → App calls ThreeScene.updateScene(params, gpu)
    → Three.js scene updates in real time
```

### `context/PhysicsContext.jsx` — Shared State

React Context with reducer pattern. Two action types:
- `SET_PARAM` — User changed a slider (lean, mass, temperature, etc.)
- `SET_GPU` — New state arrived from the server (speed, forces, penetration, etc.)

All components read from this context via `usePhysics()`.

### `hooks/useWebSocket.js` — Server Connection

Manages the WebSocket lifecycle:
- Connects to `ws://<host>/ws` on mount
- Auto-reconnects every 2 seconds on disconnect
- `sendAllParams()` called on connect to sync current slider state
- `send(obj)` for individual commands
- Dispatches incoming state to PhysicsContext

### `components/Panel/` — Input Controls

Each file is a self-contained input group:

| File | Controls |
|------|----------|
| `CenterOfMass.jsx` | G.x, G.y, G.z position + velocity inputs |
| `BladeInputs.jsx` | Foot opening angle (alpha), lean, pitch |
| `ContactPosition.jsx` | P.x, P.y (contact point on ice) |
| `SkaterInputs.jsx` | Mass slider |
| `IceInputs.jsx` | Temperature slider |
| `HollowGrind.jsx` | Hollow radius slider |
| `GpuReadouts.jsx` | All GPU-computed values (forces, penetration, contact geometry) |
| `CameraButtons.jsx` | Camera preset buttons (perspective, side, front, top, blade) |
| `BladeProfileLegend.jsx` | Rocker zone colour legend |
| `SliderControl.jsx` | Reusable slider + number input component used by all the above |
| `Panel.jsx` | Container that wires all panel components together |

### `components/Scene/` — 3D Visualisation

**`ThreeScene.jsx`** (520 lines) — Imperative Three.js canvas:
- Exposed via `forwardRef` with `updateScene(params, gpu)` method
- Creates and manages all 3D objects: blade model, ice surface, force arrows (gravity, lateral, along-blade), arc indicators (lean angle, foot opening), turn radius arc, markers (P and G points), velocity arrow
- `setCamera(mode, params)` for camera presets
- No React state inside — parent calls `updateScene()` on every frame

**`BladeModel.js`** (260 lines) — Blade geometry builder:
- Bottom surface with colour-coded rocker zones (6 zones, different colours)
- TUUK holder with beveled edges and screw holes
- Orange penetration polygon (blade region below ice surface)
- Toe arrow indicator

### `utils/`

- **`bladeProfile.js`** — Computes the rocker profile curve (X positions, radii per zone, zone boundaries) used for the blade bottom surface geometry.
- **`iceHardness.js`** — `getIceHardness(tempC)` → hardness in MPa (Barnes & Tabor model).

### `constants.js` — Shared Constants

Must stay in sync with `server/config.py`:
- `SCALE = 50`
- Blade dimensions, profile zones, zone colours

---

## WebSocket Protocol

All messages are JSON. The server sends state at ~60 Hz. The client sends commands on user interaction.

**Client → Server (commands):**
```json
{"cmd": "lean",    "value": 15}
{"cmd": "pitch",   "value": 0.5}
{"cmd": "weight",  "value": 85}
{"cmd": "ice",     "value": 4.75}
{"cmd": "alpha",   "value": 30}
{"cmd": "push",    "fx": 0, "fy": 0, "force": 2}
{"cmd": "reset"}
{"cmd": "toggle_mesh"}
{"cmd": "hollow_radius", "value": 15.875}
{"cmd": "set_velocity",  "vx": 5, "vy": 0}
```

**Server → Client (state, every 4th physics frame):**
```json
{
  "type": "state",
  "pos": [x, y, z],
  "vel": [vx, vy, 0],
  "speed": 4.99,
  "pen_max_mm": 0.425,
  "pen_analytical_mm": 0.425,
  "contact_length_mm": 103.9,
  "contact_width_mm": 1.64,
  "f_lateral": 223,
  "f_along": 10,
  "Fg": 833.9,
  "F_normal": 805.4,
  "lean_actual": 0.2618,
  "collision_mode": "mesh",
  "frame": 1234,
  "n_ice": 240000,
  ...
}
```

---

## Physics Overview

See `physics-engine-doc.md` for detailed equations. Key concepts:

1. **Particle pool** — 240k particles in a 300 x 50 x 5 mm box centred on the blade. Particles are recentred around the blade each step (infinite ice illusion).

2. **Mesh collision** — Each particle queries its signed distance to the blade CAD mesh. If inside or overlapping, a penalty force pushes it out. The reaction (Newton's 3rd law) acts on the blade.

3. **Anisotropic stiffness** — Forces along the blade length are 500x weaker than lateral/vertical forces. This models the real geometry: the thin blade end face offers little resistance, while the groove walls resist strongly.

4. **Penetration equilibrium** — `H x Lc(d) x w(d) = F_normal`. Ice hardness H times contact length times contact width equals the normal force. Deeper penetration → more contact area → more resistance → natural equilibrium.

5. **Arc turning** — Lean angle + rocker zone radius determine the turn radius: `R = R_rocker / lean_factor`. The blade follows a circular arc, updating yaw each step.

---

## Deployment

The server runs on a RunPod GPU instance. To deploy:

```bash
# On RunPod
cd /workspace
python3 warp_server.py
# Server listens on port 8765
# Sandbox available at http://<host>:8765/sandbox
```

The React build (`sandbox-ui/dist/`) must be present on the server. To update:
```bash
cd sandbox-ui && npm run build
# Copy dist/ to server, restart warp_server.py
```

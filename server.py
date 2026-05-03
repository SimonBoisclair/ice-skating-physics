"""
Path A GPU: Real Genesis physics with MPM deformable ice.
Blade is a rigid body. Ice is deformable MPM material.
Friction emerges from physical contact — blade edge cuts into ice.
WebSocket bridge sends state to Three.js browser frontend.
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import asyncio
import json
import math
import time
import multiprocessing as mp
from aiohttp import web

# ── Physical Constants ──
BLADE_LEN = 0.28       # 280mm
BLADE_T = 0.003        # 3mm thick
BLADE_H = 0.03         # 30mm tall
G = 9.81
DT = 2e-3              # 2ms timestep — interactive on GPU with MPM

# Rocker zones (for HUD display)
PROFILE_ZONES = [0.08, 0.18, 0.24, 0.24, 0.18, 0.08]
PROFILE_RADII_FT = [0.15, 15, 12, 9, 6, 0.15]
ZONE_NAMES = ['Heel Tip', "Zone 4 (15')", "Zone 3 (12')", "Zone 2 (9')", "Zone 1 (6')", 'Toe Tip']
ZONE_BOUNDS = [0]
for f in PROFILE_ZONES:
    ZONE_BOUNDS.append(ZONE_BOUNDS[-1] + f)

def get_zone_at_pitch(pitch):
    ct = 0.5 + pitch * 0.35
    for z in range(len(PROFILE_RADII_FT)):
        if ct <= ZONE_BOUNDS[z + 1] + 0.001:
            return z, ZONE_NAMES[z], PROFILE_RADII_FT[z] * 0.3048
    return len(PROFILE_RADII_FT) - 1, ZONE_NAMES[-1], PROFILE_RADII_FT[-1] * 0.3048


def to_np(x):
    """Convert Genesis tensor or numpy value to flat numpy array."""
    if hasattr(x, 'cpu'):
        return x.cpu().numpy().flatten()
    return np.array(x).flatten()


def genesis_worker(state_pipe, cmd_pipe):
    """
    Genesis GPU physics process with MPM deformable ice.
    
    The blade (rigid body) sits on deformable ice (MPM elastic material).
    When the blade is pushed, friction forces emerge naturally from the
    blade edge cutting into the ice surface. No hand-tuned friction coefficients.
    
    We still handle:
    - Push impulses (external force on blade)
    - Lean angle (orientation control)
    - Pitch (shifts blade contact point along rocker)
    
    Genesis handles:
    - Gravity
    - Deformable ice contact (MPM)
    - Friction forces from physical contact geometry
    - Blade settling and penetration
    """
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')

    # ── Scene setup ──
    scene = gs.Scene(
        show_viewer=False,
        mpm_options=gs.options.MPMOptions(
            dt=DT,
            particle_size=0.005,    # 5mm particles — ~1000+ particles, interactive
            lower_bound=(-0.5, -0.5, -0.05),   # ensure ice particles fit
            upper_bound=(0.5, 0.5, 0.5),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=DT,
            gravity=(0, 0, -G),
            enable_collision=True,
            enable_joint_limit=False,
        ),
    )

    # ── Ice surface: deformable MPM material ──
    # Ice properties: density 917 kg/m3, moderate stiffness
    # The ice patch is where the blade contacts — sized for the contact area
    ice_patch = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            rho=917,        # ice density kg/m3
            E=5e6,          # Young's modulus (stiffer to prevent deep sinking)
            nu=0.33,        # Poisson's ratio
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.01),    # thin layer at ground level
            size=(0.30, 0.08, 0.015), # 300mm x 80mm x 15mm patch (wider for sliding)
        ),
    )

    # ── Static ground plane (beneath ice to prevent particles falling) ──
    ground = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.5),
    )

    # ── Blade: rigid body ──
    # Set density so blade mass = skater mass. This lets Genesis handle weight naturally.
    # Blade volume = BLADE_LEN * BLADE_T * BLADE_H = 0.28 * 0.003 * 0.03 = 2.52e-5 m3
    # For 85kg skater: rho = 85 / 2.52e-5 ≈ 3,373,016 kg/m3
    # Use moderate mass — full skater weight causes too deep sinking at current resolution
    skater_mass = 20.0  # reduced for better MPM interaction
    blade_vol = BLADE_LEN * BLADE_T * BLADE_H
    blade_rho = skater_mass / blade_vol
    blade = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0, 0, 0.03),   # start above ice, will settle
            size=(BLADE_LEN / 2, BLADE_T / 2, BLADE_H / 2),
        ),
        material=gs.materials.Rigid(rho=blade_rho, friction=0.8),
    )

    scene.build()

    n_particles = ice_patch.n_particles
    print(f"[genesis-gpu] MPM ice patch: {n_particles} particles")
    print(f"[genesis-gpu] Timestep: {DT}s, GPU backend")

    # Let system settle (blade drops onto ice under gravity)
    print("[genesis-gpu] Settling blade onto ice...")
    for i in range(100):
        scene.step()
    print("[genesis-gpu] Settling complete")

    # Get initial state
    bm = blade.get_mass()
    blade_mass = float(to_np(bm)[0]) if len(to_np(bm)) > 0 else float(bm)
    blade_mass = max(blade_mass, 0.01)
    print(f"[genesis-gpu] Blade mass: {blade_mass:.4f} kg")

    # State variables
    lean = 15.0         # degrees
    weight = 85.0       # kg (applied as downward force)
    pitch = 0.0         # -1 to +1
    ice_mpa = 7.0

    push_fx_world = 0.0
    push_fy_world = 0.0
    push_frames = 0
    push_force_mult = 1.5

    # We'll track yaw ourselves since we directly control orientation
    sim_yaw = 0.0
    prev_pos = to_np(blade.get_pos())

    state_pipe.send({'type': 'ready'})
    print(f"[genesis-gpu] Ready! Running physics loop...")

    frame = 0
    fps_t0 = time.time()
    fps_count = 0

    while True:
        # Process commands from browser
        while cmd_pipe.poll():
            try:
                c = cmd_pipe.recv()
                cmd = c.get('cmd')
                if cmd == 'push':
                    fx_local = c.get('fx', 0)
                    fy_local = c.get('fy', 0)
                    push_force_mult = c.get('force', 1.5)
                    # Convert local push to world frame using current yaw
                    cos_y = math.cos(sim_yaw)
                    sin_y = math.sin(sim_yaw)
                    push_fx_world = fx_local * cos_y - fy_local * sin_y
                    push_fy_world = fx_local * sin_y + fy_local * cos_y
                    push_frames = 15
                elif cmd == 'lean':
                    lean = float(c.get('value', 15))
                elif cmd == 'weight':
                    weight = float(c.get('value', 85))
                elif cmd == 'pitch':
                    pitch = float(c.get('value', 0))
                elif cmd == 'ice':
                    ice_mpa = float(c.get('value', 7))
                elif cmd == 'reset':
                    sim_yaw = 0.0
                    push_frames = 0
                    blade.set_pos(np.array([0, 0, 0.03], dtype=np.float64))
                    blade.set_quat(np.array([1, 0, 0, 0], dtype=np.float64))
                    blade.set_dofs_velocity(np.zeros(6, dtype=np.float64))
                    # Re-settle
                    for _ in range(50):
                        scene.step()
            except Exception as e:
                print(f"[genesis-gpu] cmd error: {e}")

        # ── Apply forces ──
        cur_pos = to_np(blade.get_pos())
        cur_vel = to_np(blade.get_vel())

        # Push force only — Genesis handles weight via gravity on the heavy blade
        fx, fy = 0.0, 0.0
        if push_frames > 0:
            F = 500.0 * push_force_mult  # stronger push to overcome MPM friction
            fx += push_fx_world * F
            fy += push_fy_world * F
            push_frames -= 1

        # Apply push as velocity change
        dvx = fx / max(blade_mass, 0.01) * DT
        dvy = fy / max(blade_mass, 0.01) * DT

        new_vx = cur_vel[0] + dvx
        new_vy = cur_vel[1] + dvy
        new_vz = cur_vel[2]  # let Genesis handle vertical

        # Set lean angle (orientation control)
        lean_rad = lean * math.pi / 180
        # Yaw is user-controlled or stays fixed — blade heading != velocity direction
        # (For skating, you glide along blade axis but can also slide sideways)

        # Build quaternion: yaw (around Z) * lean (around local X)
        qw_yaw = math.cos(sim_yaw / 2)
        qz_yaw = math.sin(sim_yaw / 2)
        qw_lean = math.cos(lean_rad / 2)
        qx_lean = math.sin(lean_rad / 2)
        w = qw_yaw * qw_lean
        x = qw_yaw * qx_lean
        y = qz_yaw * qx_lean
        z = qz_yaw * qw_lean
        blade.set_quat(np.array([w, x, y, z], dtype=np.float64))

        # Set velocity (horizontal from us + vertical from Genesis)
        blade.set_dofs_velocity(np.array([
            new_vx, new_vy, new_vz,
            0, 0, 0
        ], dtype=np.float64))

        # Step Genesis (handles MPM ice deformation + rigid body + gravity + contact)
        scene.step()

        # Read post-step state
        pos_after = to_np(blade.get_pos())
        vel_after = to_np(blade.get_vel())

        # Dampen very low velocities to prevent residual drift
        hspeed = math.sqrt(vel_after[0]**2 + vel_after[1]**2)
        if hspeed < 0.005 and push_frames == 0:
            blade.set_dofs_velocity(np.array([
                0, 0, vel_after[2] if len(vel_after) > 2 else 0,
                0, 0, 0
            ], dtype=np.float64))
            vel_after = to_np(blade.get_vel())

        # Compute derived quantities for HUD
        speed_after = math.sqrt(vel_after[0]**2 + vel_after[1]**2)
        bx, by = math.cos(sim_yaw), math.sin(sim_yaw)
        ppx, ppy = -by, bx
        va = vel_after[0] * bx + vel_after[1] * by
        vp = vel_after[0] * ppx + vel_after[1] * ppy

        # Contact info from Genesis
        try:
            contact_force = to_np(blade.get_links_net_contact_force())
            contact_z = float(contact_force[2]) if len(contact_force) > 2 else 0
        except Exception:
            contact_z = 0

        # Estimate penetration: blade center is at pos_after[2], bottom edge is BLADE_H/2 below
        blade_bottom = pos_after[2] - BLADE_H / 2
        ice_surface = 0.0175  # top of ice patch (15mm thick centered at z=0.01)
        pen = max(0, ice_surface - blade_bottom)

        # Zone info from pitch
        zone_idx, zone_name, R = get_zone_at_pitch(pitch)

        # FPS tracking
        fps_count += 1
        if fps_count % 100 == 0:
            elapsed = time.time() - fps_t0
            fps = fps_count / elapsed if elapsed > 0 else 0
            print(f"[genesis-gpu] {fps:.0f} SPS, speed={speed_after:.3f} m/s, "
                  f"pos=({pos_after[0]:.3f}, {pos_after[1]:.3f}, {pos_after[2]:.4f})")
            fps_count = 0
            fps_t0 = time.time()

        # Send state to browser (every 4th frame ≈ 125 Hz physics / 4 = ~30 Hz updates)
        if frame % 4 == 0:
            state = {
                'type': 'state',
                'pos': [round(float(pos_after[0]), 6), round(float(pos_after[1]), 6), round(float(pos_after[2]), 6)],
                'speed': round(float(speed_after), 6),
                'va': round(float(va), 6),
                'vp': round(float(vp), 6),
                'yaw': round(float(sim_yaw), 6),
                'lean_actual': round(float(lean_rad), 6),
                'pitch_actual': round(float(pitch * 5.0 * math.pi / 180), 6),
                'mu_a': 0.0,   # no computed friction — it's emergent from physics!
                'mu_p': 0.0,
                'pen': round(float(pen), 6),
                'Lc': 0.0,
                'R': round(float(R), 4),
                'zone': int(zone_idx),
                'zone_name': zone_name,
                'contact_z': round(float(contact_z), 6),
                'frame': int(frame),
                'engine': 'genesis-gpu-mpm',
            }
            state_pipe.send(state)

        frame += 1
        prev_pos = pos_after.copy()


# ── Web Server ──
clients = set()


async def websocket_handler(request):
    global clients
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    print(f"[ws] Client connected ({len(clients)} total)")
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                request.app['cmd_pipe'].send(data)
    finally:
        clients.discard(ws)
        print(f"[ws] Client disconnected ({len(clients)} total)")
    return ws


async def index_handler(request):
    return web.FileResponse('./index.html')


async def genesis_reader(app):
    """Read state from Genesis process and broadcast to all WebSocket clients."""
    global clients
    pipe = app['state_pipe']
    loop = asyncio.get_event_loop()
    msg_count = 0
    while True:
        try:
            has_data = await loop.run_in_executor(None, pipe.poll, 0.05)
            if has_data:
                state = pipe.recv()
                if state.get('type') == 'ready':
                    print("[reader] Genesis GPU engine ready signal received")
                    continue
                msg_count += 1
                if msg_count <= 3 or msg_count % 100 == 0:
                    print(f"[reader] Msg #{msg_count}, clients={len(clients)}, "
                          f"speed={state.get('speed', '?')}")
                msg = json.dumps(state)
                dead = set()
                for ws in clients:
                    try:
                        await ws.send_str(msg)
                    except Exception:
                        dead.add(ws)
                clients -= dead
            else:
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[reader] Error: {e}")
            await asyncio.sleep(0.1)


async def start_background_tasks(app):
    app['genesis_task'] = asyncio.create_task(genesis_reader(app))


async def cleanup_background_tasks(app):
    app['genesis_task'].cancel()
    app['genesis_proc'].terminate()


def main():
    state_parent, state_child = mp.Pipe()
    cmd_parent, cmd_child = mp.Pipe()

    proc = mp.Process(target=genesis_worker, args=(state_child, cmd_child), daemon=True)
    proc.start()
    print("[server] Genesis GPU process started")

    # Wait for ready signal
    while True:
        if state_parent.poll(30.0):
            msg = state_parent.recv()
            if msg.get('type') == 'ready':
                print("[server] Genesis GPU ready, starting web server")
                break
        else:
            print("[server] Waiting for Genesis GPU to initialize...")

    app = web.Application()
    app['state_pipe'] = state_parent
    app['cmd_pipe'] = cmd_parent
    app['genesis_proc'] = proc

    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    print("[server] Starting on http://0.0.0.0:8765")
    web.run_app(app, host='0.0.0.0', port=8765, print=None)


if __name__ == '__main__':
    main()

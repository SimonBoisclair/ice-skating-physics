"""
HTTP routes, WebSocket handler, and the main physics loop.

Usage:
    from server.web import run_server
    run_server()          # blocks forever on port 8765
"""
import asyncio
import json
import os
import time

import aiohttp
from aiohttp import web
import numpy as np

from .config import SCALE, BLADE_LEN, BLADE_W, BLADE_H, N_ICE, ICE_H
from .physics import BladePhysics

# ── global state ──────────────────────────────────────────────────
physics: BladePhysics | None = None
clients: set[web.WebSocketResponse] = set()


# ── physics loop ──────────────────────────────────────────────────

async def physics_loop():
    global physics
    physics = BladePhysics()
    print(f"[server] Physics ready. {N_ICE} ice particles, {SCALE}x scale")

    frame = 0
    t0 = time.time()
    while True:
        state = physics.step()
        frame += 1

        if frame % 4 == 0 and clients:
            msg = json.dumps(state)
            dead = []
            for ws in clients:
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

        if frame % 500 == 0:
            elapsed = time.time() - t0
            sps = 500 / elapsed
            t0 = time.time()
            s  = state['speed']
            fl = state.get('f_lateral', 0)
            fa = state.get('f_along', 0)
            print(f"[physics] {sps:.0f} SPS, speed={s:.3f} m/s, "
                  f"F_lat={fl:.0f}N F_along={fa:.0f}N, clients={len(clients)}")

        await asyncio.sleep(0)


# ── WebSocket handler ─────────────────────────────────────────────

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    print(f"[ws] Client connected ({len(clients)} total)")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                    physics.handle_command(cmd)
                except Exception as e:
                    print(f"[ws] Bad command: {e}")
    finally:
        clients.discard(ws)
        print(f"[ws] Client disconnected ({len(clients)} total)")

    return ws


# ── HTTP route handlers ───────────────────────────────────────────

async def index_handler(request):
    return web.FileResponse('./index.html')


async def test_handler(request):
    return web.FileResponse('./test_live.html')


async def dashboard_handler(request):
    return web.FileResponse('./dashboard.html')


async def sandbox_handler(request):
    dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'sandbox-ui', 'dist', 'index.html')
    if os.path.exists(dist):
        return web.FileResponse(dist)
    return web.FileResponse('./sandbox.html')


async def debug_mesh_handler(request):
    """Return mesh + nearby particle data as JSON for debug visualisation."""
    if physics is None:
        return web.json_response({'error': 'Physics not initialized'}, status=503)

    ice_np = physics.ice_pos.numpy()
    cx, cy, cz = physics.pos[0], physics.pos[1], physics.pos[2]
    half_l = BLADE_LEN / 2.0
    half_w = BLADE_W * 6.0

    mask = (
        (np.abs(ice_np[:, 0] - cx) < half_l * 1.5) &
        (np.abs(ice_np[:, 1] - cy) < half_w) &
        (ice_np[:, 2] < BLADE_H * 1.5)
    )
    near_particles = ice_np[mask]

    response = {
        'blade_pos': [float(cx), float(cy), float(cz)],
        'blade_yaw': float(physics.yaw + physics.alpha),
        'blade_lean': float(physics.lean),
        'collision_mode': 'mesh' if physics.blade_mesh is not None else 'box',
        'hollow_radius_mm': round(physics.hollow_radius * 1000, 2),
        'n_particles_near': int(mask.sum()),
        'n_particles_total': N_ICE,
        'particles': near_particles.tolist(),
        'scale': SCALE,
    }

    if physics.mesh_data is not None:
        response['mesh'] = physics.mesh_data

    return web.json_response(response)


# ── lifecycle hooks ───────────────────────────────────────────────

async def start_background_tasks(app):
    app['physics_task'] = asyncio.ensure_future(physics_loop())


async def cleanup_background_tasks(app):
    app['physics_task'].cancel()


# ── entry point ───────────────────────────────────────────────────

def run_server(host='0.0.0.0', port=8765):
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/test', test_handler)
    app.router.add_get('/dashboard', dashboard_handler)
    app.router.add_get('/sandbox', sandbox_handler)
    app.router.add_get('/debug_mesh', debug_mesh_handler)
    app.router.add_get('/ws', ws_handler)

    # Serve React build static assets
    dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'sandbox-ui', 'dist')
    if os.path.isdir(dist_dir):
        app.router.add_static('/assets', os.path.join(dist_dir, 'assets'))

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    print(f"[server] Starting on port {port}...")
    web.run_app(app, host=host, port=port)

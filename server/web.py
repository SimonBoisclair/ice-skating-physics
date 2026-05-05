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
# Try Warp 3D renderer first, fall back to PIL renderer
try:
    from .renderer_warp import WarpParticleRenderer as ParticleRenderer
    print("[server] Using Warp OpenGL 3D renderer")
except Exception as e:
    from .renderer import ParticleRenderer
    print(f"[server] Warp renderer unavailable ({e}), using PIL 2D renderer")

# ── global state ──────────────────────────────────────────────────
physics: BladePhysics | None = None
clients: set[web.WebSocketResponse] = set()
renderer = None
stream_clients: set[web.StreamResponse] = set()


# ── physics loop ──────────────────────────────────────────────────

async def physics_loop():
    global physics
    physics = BladePhysics()
    global renderer
    renderer = ParticleRenderer()
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

        # Render MJPEG frame every 8th physics step (~30fps if physics is 250Hz)
        if frame % 8 == 0 and stream_clients:
            try:
                jpeg_bytes = renderer.render_frame(physics)
                dead_streams = []
                for resp in stream_clients:
                    try:
                        await resp.write(
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n'
                            b'\r\n' + jpeg_bytes + b'\r\n'
                        )
                    except Exception:
                        dead_streams.append(resp)
                for resp in dead_streams:
                    stream_clients.discard(resp)
            except Exception as e:
                if frame % 200 == 0:
                    print(f"[stream] Render error: {e}")

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

async def sandbox_handler(request):
    dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'sandbox-ui', 'dist', 'index.html')
    return web.FileResponse(dist)


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


async def stream_handler(request):
    """MJPEG stream endpoint. Streams particle visualization as video."""
    if physics is None:
        return web.Response(text='Physics not initialized', status=503)

    resp = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await resp.prepare(request)
    stream_clients.add(resp)
    print(f"[stream] Client connected ({len(stream_clients)} total)")

    try:
        # Keep connection alive until client disconnects
        while True:
            await asyncio.sleep(1)
            if resp.task is None or resp.task.done():
                break
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        stream_clients.discard(resp)
        print(f"[stream] Client disconnected ({len(stream_clients)} total)")

    return resp


async def viz_handler(request):
    """Serve the particle visualization page."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GPU Particle Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a14;
            color: #c8d0dc;
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #e0e4ec;
        }
        .info {
            font-size: 12px;
            color: #8890a0;
            margin-bottom: 15px;
        }
        .stream-container {
            border: 1px solid #2a3040;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        img {
            display: block;
            width: 1280px;
            height: 720px;
            max-width: 95vw;
            height: auto;
            image-rendering: pixelated;
        }
        .status {
            margin-top: 10px;
            font-size: 11px;
            color: #6a7080;
        }
        .status.connected { color: #4a9; }
        .controls {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        button {
            background: #1a2030;
            color: #c8d0dc;
            border: 1px solid #2a3040;
            padding: 8px 20px;
            border-radius: 4px;
            font-family: inherit;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.15s;
        }
        button:hover { background: #252d40; }
        button.play {
            background: #1a3a25;
            border-color: #2a5a3a;
            color: #6fdb8f;
            font-size: 14px;
            padding: 10px 28px;
        }
        button.play:hover { background: #254a32; }
        button.stop {
            background: #3a1a1a;
            border-color: #5a2a2a;
            color: #db6f6f;
        }
        button.stop:hover { background: #4a2525; }
        button.reset {
            background: #1a2a3a;
            border-color: #2a3a5a;
            color: #6faadb;
        }
        button.reset:hover { background: #253550; }
        button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        a { color: #5a8abf; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .sim-controls {
            margin-top: 15px;
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .sim-status {
            font-size: 11px;
            color: #6a7080;
            margin-left: 10px;
        }
        .sim-status.running { color: #6fdb8f; }
        .sim-status.paused { color: #dba86f; }
        .bottom-controls {
            margin-top: 10px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
    </style>
</head>
<body>
    <h1>GPU Particle Physics Stream</h1>
    <p class="info">Live MJPEG stream of 240k ice particles from the GPU simulation</p>
    <div class="stream-container">
        <img id="stream" src="/stream" alt="Particle stream loading..." />
    </div>
    <div class="sim-controls">
        <button class="play" id="btnPlay" onclick="startSim()">&#9654; Play</button>
        <button class="stop" id="btnPause" onclick="pauseSim()" disabled>&#9632; Pause</button>
        <button class="reset" id="btnReset" onclick="resetSim()">&#8634; Reset</button>
        <span class="sim-status paused" id="simStatus">Paused</span>
    </div>
    <p class="status" id="status">Connecting...</p>
    <div class="bottom-controls">
        <button onclick="document.getElementById('stream').src='/stream?t='+Date.now()">Reconnect Stream</button>
        <a href="/sandbox">Open Sandbox</a>
    </div>
    <script>
        const img = document.getElementById('stream');
        const status = document.getElementById('status');
        const simStatus = document.getElementById('simStatus');
        const btnPlay = document.getElementById('btnPlay');
        const btnPause = document.getElementById('btnPause');
        let ws = null;
        let isRunning = false;

        function connectWs() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');
            ws.onopen = () => {
                status.textContent = 'Connected - streaming';
                status.className = 'status connected';
            };
            ws.onclose = () => {
                status.textContent = 'WebSocket disconnected - reconnecting...';
                status.className = 'status';
                setTimeout(connectWs, 2000);
            };
            ws.onmessage = (e) => {
                try {
                    const state = JSON.parse(e.data);
                    if (state.physics_paused !== undefined) {
                        isRunning = !state.physics_paused;
                        updateButtons();
                    }
                } catch(err) {}
            };
        }

        function send(cmd) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(cmd));
            }
        }

        function startSim() {
            send({cmd: 'reset_blade_position'});
            send({cmd: 'start_penetration'});
            isRunning = true;
            updateButtons();
        }

        function pauseSim() {
            send({cmd: 'pause'});
            isRunning = false;
            updateButtons();
        }

        function resetSim() {
            send({cmd: 'reset'});
            isRunning = false;
            updateButtons();
        }

        function updateButtons() {
            btnPlay.disabled = isRunning;
            btnPause.disabled = !isRunning;
            simStatus.textContent = isRunning ? 'Running' : 'Paused';
            simStatus.className = 'sim-status ' + (isRunning ? 'running' : 'paused');
        }

        img.onload = () => {
            if (!status.className.includes('connected')) {
                status.textContent = 'Connected - streaming';
                status.className = 'status connected';
            }
        };
        img.onerror = () => {
            status.textContent = 'Stream disconnected - click Reconnect';
            status.className = 'status';
        };

        connectWs();
    </script>
</body>
</html>
"""
    return web.Response(text=html, content_type='text/html')


# ── lifecycle hooks ───────────────────────────────────────────────

async def start_background_tasks(app):
    app['physics_task'] = asyncio.ensure_future(physics_loop())


async def cleanup_background_tasks(app):
    app['physics_task'].cancel()


# ── entry point ───────────────────────────────────────────────────

def run_server(host='0.0.0.0', port=8765):
    app = web.Application()
    app.router.add_get('/', sandbox_handler)
    app.router.add_get('/sandbox', sandbox_handler)
    app.router.add_get('/debug_mesh', debug_mesh_handler)
    app.router.add_get('/stream', stream_handler)
    app.router.add_get('/viz', viz_handler)
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

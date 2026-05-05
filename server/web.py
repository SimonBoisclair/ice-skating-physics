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
                    # Camera commands go to renderer, physics commands to physics
                    if cmd.get('cmd') == 'camera' and renderer is not None:
                        if 'azimuth' in cmd:
                            renderer.cam_azimuth = float(cmd['azimuth'])
                        if 'elevation' in cmd:
                            renderer.cam_elevation = float(cmd['elevation'])
                        if 'distance' in cmd:
                            renderer.cam_distance = float(cmd['distance'])
                    else:
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
    """Serve the particle visualization page with interactive camera controls."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>GPU Particle Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a14;
            color: #c8d0dc;
            font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', 'Menlo', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            min-height: 100dvh;
            padding: 10px;
            overflow-x: hidden;
            touch-action: none;
        }
        h1 { font-size: 16px; font-weight: 600; margin-bottom: 6px; color: #e0e4ec; }
        .info { font-size: 11px; color: #8890a0; margin-bottom: 10px; text-align: center; }
        .stream-container {
            border: 1px solid #2a3040;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            position: relative;
            cursor: grab;
            touch-action: none;
            width: 100%;
            max-width: 1280px;
        }
        .stream-container:active { cursor: grabbing; }
        #stream {
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 16/9;
            image-rendering: auto;
            pointer-events: none;
            -webkit-user-drag: none;
            user-select: none;
        }
        .cam-hint {
            position: absolute;
            bottom: 8px;
            right: 8px;
            font-size: 10px;
            color: rgba(200,210,220,0.6);
            background: rgba(10,10,20,0.7);
            padding: 3px 8px;
            border-radius: 4px;
            pointer-events: none;
        }
        .controls-row {
            margin-top: 10px;
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            background: #1a2030;
            color: #c8d0dc;
            border: 1px solid #2a3040;
            padding: 10px 18px;
            border-radius: 6px;
            font-family: inherit;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.15s;
            -webkit-tap-highlight-color: transparent;
        }
        button:hover { background: #252d40; }
        button:active { background: #303848; }
        button.play { background: #1a3a25; border-color: #2a5a3a; color: #6fdb8f; }
        button.play:hover { background: #254a32; }
        button.stop { background: #3a1a1a; border-color: #5a2a2a; color: #db6f6f; }
        button.stop:hover { background: #4a2525; }
        button.reset { background: #1a2a3a; border-color: #2a3a5a; color: #6faadb; }
        button.reset:hover { background: #253550; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .sim-status { font-size: 11px; color: #6a7080; margin-left: 6px; }
        .sim-status.running { color: #6fdb8f; }
        .sim-status.paused { color: #dba86f; }
        .status { margin-top: 6px; font-size: 10px; color: #6a7080; }
        .status.connected { color: #4a9; }
        a { color: #5a8abf; text-decoration: none; font-size: 12px; }
        a:hover { text-decoration: underline; }
        @media (max-width: 600px) {
            body { padding: 5px; }
            h1 { font-size: 14px; }
            .info { font-size: 10px; margin-bottom: 6px; }
            button { padding: 12px 14px; font-size: 14px; }
        }
    </style>
</head>
<body>
    <h1>GPU Particle Physics Stream</h1>
    <p class="info">Drag to orbit &bull; Scroll/pinch to zoom &bull; 240k particles live</p>
    <div class="stream-container" id="viewport">
        <img id="stream" src="/stream" alt="Loading..." />
        <div class="cam-hint" id="camHint">Drag to orbit</div>
    </div>
    <div class="controls-row">
        <button class="play" id="btnPlay" onclick="startSim()">&#9654; Play</button>
        <button class="stop" id="btnPause" onclick="pauseSim()" disabled>&#9632; Pause</button>
        <button class="reset" id="btnReset" onclick="resetSim()">&#8634; Reset</button>
        <span class="sim-status paused" id="simStatus">Paused</span>
    </div>
    <div class="controls-row">
        <button onclick="document.getElementById('stream').src='/stream?t='+Date.now()">Reconnect</button>
        <a href="/sandbox">Open Sandbox</a>
    </div>
    <p class="status" id="status">Connecting...</p>
    <script>
        const img = document.getElementById('stream');
        const viewport = document.getElementById('viewport');
        const status = document.getElementById('status');
        const simStatus = document.getElementById('simStatus');
        const btnPlay = document.getElementById('btnPlay');
        const btnPause = document.getElementById('btnPause');
        const camHint = document.getElementById('camHint');
        let ws = null;
        let isRunning = false;

        // Camera state (orbit)
        let camAz = 0.4;       // azimuth (radians)
        let camEl = 0.6;       // elevation (radians)
        let camDist = """ + str(BLADE_LEN * 1.5) + """;  // distance

        const MIN_EL = 0.05;
        const MAX_EL = 1.5;
        const MIN_DIST = """ + str(BLADE_LEN * 0.3) + """;
        const MAX_DIST = """ + str(BLADE_LEN * 5.0) + """;

        function sendCamera() {
            send({cmd: 'camera', azimuth: camAz, elevation: camEl, distance: camDist});
        }

        // ─── Mouse controls ───
        let dragging = false;
        let lastX = 0, lastY = 0;

        viewport.addEventListener('mousedown', (e) => {
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            e.preventDefault();
        });
        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            camAz += dx * 0.005;
            camEl = Math.max(MIN_EL, Math.min(MAX_EL, camEl + dy * 0.005));
            sendCamera();
        });
        window.addEventListener('mouseup', () => { dragging = false; });

        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            camDist = Math.max(MIN_DIST, Math.min(MAX_DIST, camDist * (1 + e.deltaY * 0.001)));
            sendCamera();
        }, {passive: false});

        // ─── Touch controls ───
        let touches = {};
        let lastPinchDist = 0;

        viewport.addEventListener('touchstart', (e) => {
            e.preventDefault();
            camHint.style.display = 'none';
            for (const t of e.changedTouches) {
                touches[t.identifier] = {x: t.clientX, y: t.clientY};
            }
            if (e.touches.length === 2) {
                const [a, b] = [e.touches[0], e.touches[1]];
                lastPinchDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
            }
        }, {passive: false});

        viewport.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                // Single finger: orbit
                const t = e.touches[0];
                const prev = touches[t.identifier];
                if (prev) {
                    const dx = t.clientX - prev.x;
                    const dy = t.clientY - prev.y;
                    camAz += dx * 0.006;
                    camEl = Math.max(MIN_EL, Math.min(MAX_EL, camEl + dy * 0.006));
                    sendCamera();
                }
                touches[t.identifier] = {x: t.clientX, y: t.clientY};
            } else if (e.touches.length === 2) {
                // Pinch: zoom
                const [a, b] = [e.touches[0], e.touches[1]];
                const dist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
                if (lastPinchDist > 0) {
                    const scale = lastPinchDist / dist;
                    camDist = Math.max(MIN_DIST, Math.min(MAX_DIST, camDist * scale));
                    sendCamera();
                }
                lastPinchDist = dist;
                for (const t of e.changedTouches) {
                    touches[t.identifier] = {x: t.clientX, y: t.clientY};
                }
            }
        }, {passive: false});

        viewport.addEventListener('touchend', (e) => {
            for (const t of e.changedTouches) {
                delete touches[t.identifier];
            }
            lastPinchDist = 0;
        });

        // ─── WebSocket ───
        function connectWs() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');
            ws.onopen = () => {
                status.textContent = 'Connected';
                status.className = 'status connected';
                sendCamera();  // sync camera state on connect
            };
            ws.onclose = () => {
                status.textContent = 'Reconnecting...';
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
                status.textContent = 'Connected';
                status.className = 'status connected';
            }
        };
        img.onerror = () => {
            status.textContent = 'Stream lost - click Reconnect';
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

"""
HTTP routes, WebSocket handler, and the main physics loop.
"""
import asyncio
import json
import os
import subprocess
import time

import aiohttp
from aiohttp import web

from .config import N_ICE
from .particle_sim import ParticlePoolSimulation
from .renderer_warp import WarpParticleRenderer

physics: ParticlePoolSimulation | None = None
clients: set[web.WebSocketResponse] = set()
renderer = None
stream_clients: set[web.StreamResponse] = set()

IDLE_TIMEOUT_SECS = 10 * 60
last_activity = time.time()


def touch_activity():
    global last_activity
    last_activity = time.time()


async def physics_loop():
    global physics, renderer
    physics = ParticlePoolSimulation()
    renderer = WarpParticleRenderer()
    print(f"[server] Ready. {N_ICE} particles")

    frame = 0
    t0 = time.time()
    while True:
        physics.step()
        frame += 1

        if frame % 8 == 0 and stream_clients:
            try:
                jpeg_bytes = renderer.render_frame(physics)
                dead = []
                for resp in stream_clients:
                    try:
                        await resp.write(
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(jpeg_bytes)).encode() + b'\r\n'
                            b'\r\n' + jpeg_bytes + b'\r\n'
                        )
                    except Exception:
                        dead.append(resp)
                for resp in dead:
                    stream_clients.discard(resp)
            except Exception as e:
                if frame % 200 == 0:
                    print(f"[stream] Render error: {e}")

        if frame % 500 == 0:
            elapsed = time.time() - t0
            sps = 500 / elapsed
            t0 = time.time()
            print(f"[physics] {sps:.0f} SPS, clients={len(clients)}")

        await asyncio.sleep(0)


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    touch_activity()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                    touch_activity()
                    if cmd.get('cmd') == 'camera' and renderer is not None:
                        if 'azimuth' in cmd:
                            renderer.cam_azimuth = float(cmd['azimuth'])
                        if 'elevation' in cmd:
                            renderer.cam_elevation = float(cmd['elevation'])
                        if 'distance' in cmd:
                            renderer.cam_distance = float(cmd['distance'])
                        await ws.send_str(json.dumps({
                            'ack': 'camera',
                            'az': renderer.cam_azimuth,
                            'el': renderer.cam_elevation,
                            'd': renderer.cam_distance,
                        }))
                    else:
                        physics.handle_command(cmd)
                        await ws.send_str(json.dumps({
                            'physics_paused': physics.physics_paused,
                        }))
                except Exception as e:
                    print(f"[ws] Bad command: {e}")
    finally:
        clients.discard(ws)

    return ws


async def stream_handler(request):
    """MJPEG stream endpoint."""
    if physics is None:
        return web.Response(text='Not ready', status=503)

    physics.handle_command({'cmd': 'reset_particles_only'})

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
    touch_activity()

    try:
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        stream_clients.discard(resp)

    return resp


async def viz_handler(request):
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Ice Particle Sim</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a14; color: #c8d0dc;
            font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace;
            display: flex; flex-direction: column; align-items: center;
            min-height: 100vh; padding: 10px;
            touch-action: none;
        }
        h1 { font-size: 16px; margin-bottom: 6px; }
        .info { font-size: 11px; color: #8890a0; margin-bottom: 10px; }
        .viewport {
            border: 1px solid #2a3040; border-radius: 8px; overflow: hidden;
            position: relative; cursor: grab; touch-action: none;
            width: 100%; max-width: 1280px;
        }
        .viewport:active { cursor: grabbing; }
        #stream {
            display: block; width: 100%; height: auto; aspect-ratio: 16/9;
            pointer-events: none; user-select: none;
        }
        .overlay {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            z-index: 10; touch-action: none; cursor: grab;
        }
        .overlay:active { cursor: grabbing; }
        .cam-btns {
            position: absolute; top: 10px; right: 10px; z-index: 20;
            display: grid; grid-template-columns: repeat(3, 34px); gap: 4px;
            padding: 6px; background: rgba(10,10,20,0.72);
            border: 1px solid rgba(120,140,180,0.35); border-radius: 8px;
        }
        .cam-btns button { width: 34px; height: 30px; padding: 0; font-size: 12px; }
        .controls {
            margin-top: 10px; display: flex; gap: 8px; align-items: center;
            flex-wrap: wrap; justify-content: center;
        }
        button {
            background: #1a2030; color: #c8d0dc; border: 1px solid #2a3040;
            padding: 10px 18px; border-radius: 6px; font-family: inherit;
            font-size: 13px; cursor: pointer;
        }
        button:hover { background: #252d40; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .play { background: #1a3a25; border-color: #2a5a3a; color: #6fdb8f; }
        .stop { background: #3a1a1a; border-color: #5a2a2a; color: #db6f6f; }
        .reset { background: #1a2a3a; border-color: #2a3a5a; color: #6faadb; }
        .status { margin-top: 6px; font-size: 11px; }
        .status.running { color: #6fdb8f; }
        .status.paused { color: #dba86f; }
    </style>
</head>
<body>
    <h1>Ice Particle Sim</h1>
    <p class="info">240k particles &bull; Drag to orbit &bull; Scroll to zoom</p>
    <div class="viewport" id="viewport">
        <img id="stream" alt="Loading..." />
        <div class="overlay" id="overlay"></div>
        <div class="cam-btns">
            <button onclick="moveCamera(0, 0.12)">Up</button>
            <button onclick="zoomCamera(0.85)">In</button>
            <button onclick="setCameraPreset('top')">Top</button>
            <button onclick="moveCamera(-0.18, 0)">Left</button>
            <button onclick="setCameraPreset('home')">Home</button>
            <button onclick="moveCamera(0.18, 0)">Right</button>
            <button onclick="setCameraPreset('side')">Side</button>
            <button onclick="zoomCamera(1.18)">Out</button>
            <button onclick="moveCamera(0, -0.12)">Down</button>
        </div>
    </div>
    <div class="controls">
        <button class="play" id="btnPlay" onclick="startSim()">Play</button>
        <button class="stop" id="btnPause" onclick="pauseSim()" disabled>Pause</button>
        <button class="reset" id="btnReset" onclick="resetSim()">Reset</button>
        <span class="status paused" id="simStatus">Paused</span>
    </div>
    <script>
        const img = document.getElementById('stream');
        const overlay = document.getElementById('overlay');
        const btnPlay = document.getElementById('btnPlay');
        const btnPause = document.getElementById('btnPause');
        const simStatus = document.getElementById('simStatus');
        let ws = null;
        let isRunning = false;
        let camAz = 0.076, camEl = 1.305, camDist = 21.0;

        // Stream: MJPEG on desktop, frame polling on mobile
        if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
            let busy = false;
            async function poll() {
                if (busy) return;
                busy = true;
                try {
                    const r = await fetch('/frame?t=' + Date.now());
                    if (r.ok) {
                        const blob = await r.blob();
                        const url = URL.createObjectURL(blob);
                        const old = img.src;
                        img.src = url;
                        if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
                    }
                } catch(e) {}
                busy = false;
                setTimeout(poll, 33);
            }
            poll();
        } else {
            img.src = '/stream';
        }

        function sendCamera() {
            send({cmd: 'camera', azimuth: camAz, elevation: camEl, distance: camDist});
        }
        function setCameraPreset(mode) {
            if (mode === 'top') { camAz = 1.5708; camEl = 1.56; camDist = 30; }
            else if (mode === 'home') { camAz = 1.5708; camEl = 1.2; camDist = 30; }
            else if (mode === 'side') { camAz = 1.5708; camEl = 0; camDist = 25; }
            sendCamera();
        }
        function zoomCamera(f) { camDist *= f; sendCamera(); }
        function moveCamera(dAz, dEl) { camAz += dAz; camEl += dEl; sendCamera(); }
        window.setCameraPreset = setCameraPreset;
        window.zoomCamera = zoomCamera;
        window.moveCamera = moveCamera;

        // Mouse orbit
        let dragging = false, lastX = 0, lastY = 0;
        overlay.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; e.preventDefault(); });
        window.addEventListener('mousemove', e => {
            if (!dragging) return;
            camAz -= (e.clientX - lastX) * 0.005;
            camEl -= (e.clientY - lastY) * 0.005;
            lastX = e.clientX; lastY = e.clientY;
            sendCamera();
        });
        window.addEventListener('mouseup', () => { dragging = false; });
        overlay.addEventListener('wheel', e => { e.preventDefault(); camDist *= 1 + e.deltaY * 0.001; sendCamera(); }, {passive: false});

        // Touch orbit + pinch zoom
        let touches = {}, lastPinch = 0;
        overlay.addEventListener('touchstart', e => {
            e.preventDefault();
            for (const t of e.changedTouches) touches[t.identifier] = {x: t.clientX, y: t.clientY};
            if (e.touches.length === 2) {
                const [a,b] = [e.touches[0], e.touches[1]];
                lastPinch = Math.hypot(a.clientX-b.clientX, a.clientY-b.clientY);
            }
        }, {passive: false});
        overlay.addEventListener('touchmove', e => {
            e.preventDefault();
            if (e.touches.length === 1) {
                const t = e.touches[0], prev = touches[t.identifier];
                if (prev) { camAz -= (t.clientX-prev.x)*0.006; camEl -= (t.clientY-prev.y)*0.006; sendCamera(); }
                touches[t.identifier] = {x: t.clientX, y: t.clientY};
            } else if (e.touches.length === 2) {
                const [a,b] = [e.touches[0], e.touches[1]];
                const dist = Math.hypot(a.clientX-b.clientX, a.clientY-b.clientY);
                if (lastPinch > 0) { camDist *= lastPinch/dist; sendCamera(); }
                lastPinch = dist;
                for (const t of e.changedTouches) touches[t.identifier] = {x: t.clientX, y: t.clientY};
            }
        }, {passive: false});
        overlay.addEventListener('touchend', e => {
            for (const t of e.changedTouches) delete touches[t.identifier];
            lastPinch = 0;
        });

        // WebSocket
        function connectWs() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');
            ws.onopen = () => { send({cmd: 'reset_particles_only'}); isRunning = false; updateUI(); sendCamera(); };
            ws.onclose = () => { setTimeout(connectWs, 2000); };
            ws.onmessage = e => {
                try {
                    const d = JSON.parse(e.data);
                    if (d.physics_paused !== undefined) { isRunning = !d.physics_paused; updateUI(); }
                } catch(err) {}
            };
        }
        function send(cmd) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(cmd)); }
        function startSim() { send({cmd: 'start_particles_only'}); isRunning = true; updateUI(); }
        function pauseSim() { send({cmd: 'pause'}); isRunning = false; updateUI(); }
        function resetSim() { send({cmd: 'reset_particles_only'}); isRunning = false; updateUI(); }
        function updateUI() {
            btnPlay.disabled = isRunning;
            btnPause.disabled = !isRunning;
            simStatus.textContent = isRunning ? 'Running' : 'Paused';
            simStatus.className = 'status ' + (isRunning ? 'running' : 'paused');
        }
        connectWs();
    </script>
</body>
</html>"""
    return web.Response(
        text=html,
        content_type='text/html',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
    )


async def frame_handler(request):
    """Single JPEG frame for mobile polling."""
    touch_activity()
    if physics is None or renderer is None:
        return web.Response(text='Not ready', status=503)
    try:
        jpeg_bytes = renderer.render_frame(physics)
        return web.Response(
            body=jpeg_bytes,
            content_type='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store'},
        )
    except Exception as e:
        return web.Response(text=str(e), status=500)


async def idle_watchdog():
    while True:
        await asyncio.sleep(60)
        idle_secs = time.time() - last_activity
        if len(clients) > 0 or len(stream_clients) > 0:
            touch_activity()
            continue
        if idle_secs < IDLE_TIMEOUT_SECS:
            continue
        print(f"[idle] No activity for {idle_secs/60:.0f}m — stopping pod")
        pod_id = os.environ.get('RUNPOD_POD_ID')
        api_key = os.environ.get('RUNPOD_API_KEY')
        try:
            if pod_id:
                subprocess.run(['runpodctl', 'stop', 'pod', pod_id], timeout=30, check=True)
            else:
                subprocess.run(['runpodctl', 'stop', 'pod', '--all'], timeout=30, check=True)
        except Exception:
            if pod_id and api_key:
                try:
                    import urllib.request
                    query = json.dumps({
                        'query': f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id }} }}'
                    })
                    req = urllib.request.Request(
                        'https://api.runpod.io/graphql',
                        data=query.encode(),
                        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
                    )
                    urllib.request.urlopen(req, timeout=30)
                except Exception:
                    pass
        break


async def start_background_tasks(app):
    app['physics_task'] = asyncio.ensure_future(physics_loop())
    app['idle_task'] = asyncio.ensure_future(idle_watchdog())


async def cleanup_background_tasks(app):
    app['physics_task'].cancel()
    app['idle_task'].cancel()


def run_server(host='0.0.0.0', port=8765):
    app = web.Application()
    app.router.add_get('/', viz_handler)
    app.router.add_get('/viz', viz_handler)
    app.router.add_get('/stream', stream_handler)
    app.router.add_get('/frame', frame_handler)
    app.router.add_get('/ws', ws_handler)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    print(f"[server] Starting on port {port}...")
    web.run_app(app, host=host, port=port)

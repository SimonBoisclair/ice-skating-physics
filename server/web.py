"""
HTTP routes, WebSocket handler, and the main physics loop.

Usage:
    from server.web import run_server
    run_server()          # blocks forever on port 8765
"""
import asyncio
import json
import os
import subprocess
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

# Idle timeout: stop pod after 30 minutes of no clients
IDLE_TIMEOUT_SECS = 30 * 60  # 30 minutes
last_activity = time.time()


def touch_activity():
    """Update last-activity timestamp (call on any client interaction)."""
    global last_activity
    last_activity = time.time()


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
    touch_activity()
    print(f"[ws] Client connected ({len(clients)} total)")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                    # Camera commands go to renderer, physics commands to physics
                    touch_activity()
                    if cmd.get('cmd') == 'camera' and renderer is not None:
                        if 'azimuth' in cmd:
                            renderer.cam_azimuth = float(cmd['azimuth'])
                        if 'elevation' in cmd:
                            renderer.cam_elevation = float(cmd['elevation'])
                        if 'distance' in cmd:
                            renderer.cam_distance = float(cmd['distance'])
                        print(f"[cam] az={renderer.cam_azimuth:.3f} el={renderer.cam_elevation:.3f} d={renderer.cam_distance:.1f}")
                        await ws.send_str(json.dumps({'ack': 'camera', 'az': renderer.cam_azimuth, 'el': renderer.cam_elevation, 'd': renderer.cam_distance}))
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
    touch_activity()
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


async def frame_handler(request):
    """Single JPEG frame endpoint for iOS/mobile polling."""
    touch_activity()
    if physics is None or renderer is None:
        return web.Response(text='Not ready', status=503)
    try:
        t0 = time.time()
        jpeg_bytes = renderer.render_frame(physics)
        dt = (time.time() - t0) * 1000
        print(f"[frame] render {dt:.0f}ms  cam=({renderer.cam_azimuth:.3f},{renderer.cam_elevation:.3f},{renderer.cam_distance:.1f})  size={len(jpeg_bytes)}")
        return web.Response(
            body=jpeg_bytes,
            content_type='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store', 'Pragma': 'no-cache'}
        )
    except Exception as e:
        return web.Response(text=str(e), status=500)


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
            -webkit-touch-callout: none;
            user-select: none;
        }
        .touch-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 10;
            touch-action: none;
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            user-select: none;
            cursor: grab;
        }
        .touch-overlay:active { cursor: grabbing; }
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
            z-index: 11;
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
        #debugLog {
            margin-top: 10px;
            width: 100%;
            max-width: 1280px;
            height: 200px;
            overflow-y: auto;
            background: #0d0d1a;
            border: 1px solid #2a3040;
            border-radius: 6px;
            padding: 8px;
            font-size: 11px;
            font-family: 'SF Mono', 'Menlo', monospace;
            color: #8fbc8f;
            white-space: pre-wrap;
            word-break: break-all;
        }
        @media (max-width: 600px) {
            body { padding: 5px; }
            h1 { font-size: 14px; }
            .info { font-size: 10px; margin-bottom: 6px; }
            button { padding: 12px 14px; font-size: 14px; }
            #debugLog { height: 180px; font-size: 10px; }
        }
    </style>
</head>
<body>
    <h1>GPU Particle Physics Stream</h1>
    <p class="info">Drag to orbit &bull; Scroll/pinch to zoom &bull; 240k particles live</p>
    <div class="stream-container" id="viewport">
        <img id="stream" alt="Loading..." />
        <div class="touch-overlay" id="touchOverlay"></div>
        <div class="cam-hint" id="camHint">Drag to orbit</div>
    </div>
    <div class="controls-row">
        <button class="play" id="btnPlay" onclick="startSim()">&#9654; Play</button>
        <button class="stop" id="btnPause" onclick="pauseSim()" disabled>&#9632; Pause</button>
        <button class="reset" id="btnReset" onclick="resetSim()">&#8634; Reset</button>
        <span class="sim-status paused" id="simStatus">Paused</span>
    </div>
    <div class="controls-row">
        <button id="btnReconnect" onclick="document.getElementById('stream').src='/stream?t='+Date.now()">Reconnect</button>
        <a href="/sandbox">Open Sandbox</a>
    </div>
    <p class="status" id="status">Connecting...</p>
    <div style="margin:6px 0;"><button onclick="navigator.clipboard.writeText(dbg.innerText).then(()=>this.textContent='Copied!').catch(()=>{const ta=document.createElement('textarea');ta.value=dbg.innerText;document.body.appendChild(ta);ta.select();document.execCommand('copy');document.body.removeChild(ta);this.textContent='Copied!'});setTimeout(()=>this.textContent='Copy Logs',1500)" style="padding:4px 12px;font-size:12px;background:#334;color:#aaa;border:1px solid #555;border-radius:4px;cursor:pointer">Copy Logs</button></div>
    <div id="debugLog"></div>
    <script>
        const dbg = document.getElementById('debugLog');
        function log(msg) {
            const line = document.createElement('div');
            const ts = new Date().toLocaleTimeString('en', {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit', fractionalSecondDigits:2});
            line.textContent = ts + ' ' + msg;
            dbg.appendChild(line);
            dbg.scrollTop = dbg.scrollHeight;
            if (dbg.children.length > 200) dbg.removeChild(dbg.firstChild);
        }
        log('Page loaded');

        // iOS Safari doesn't support MJPEG streams — use frame polling instead
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        const img = document.getElementById('stream');
        if (isMobile) {
            log('[stream] Mobile — fetch polling mode');
            document.getElementById('btnReconnect').style.display = 'none';
            let frameNum = 0;
            let busy = false;
            async function pollFrame() {
                if (busy) return;
                busy = true;
                try {
                    const t0 = performance.now();
                    const resp = await fetch('/frame?t=' + Date.now());
                    if (resp.ok) {
                        const blob = await resp.blob();
                        const url = URL.createObjectURL(blob);
                        const oldSrc = img.src;
                        img.src = url;
                        if (oldSrc && oldSrc.startsWith('blob:')) URL.revokeObjectURL(oldSrc);
                        frameNum++;
                        const dt = (performance.now() - t0).toFixed(0);
                        log('[stream] frame ' + frameNum + ' size=' + blob.size + ' dt=' + dt + 'ms');
                    } else {
                        log('[stream] frame resp ' + resp.status);
                    }
                } catch(e) {
                    log('[stream] error: ' + e.message);
                }
                busy = false;
                setTimeout(pollFrame, 16);
            }
            pollFrame();
        } else {
            log('[stream] Desktop — MJPEG stream');
            img.src = '/stream';
        }
        const viewport = document.getElementById('viewport');
        const overlay = document.getElementById('touchOverlay');
        const status = document.getElementById('status');
        const simStatus = document.getElementById('simStatus');
        const btnPlay = document.getElementById('btnPlay');
        const btnPause = document.getElementById('btnPause');
        const camHint = document.getElementById('camHint');
        let ws = null;
        let isRunning = false;

        // Camera state (orbit)
        let camAz = 1.5708;    // azimuth (radians, π/2 = side view)
        let camEl = 0.6;       // elevation (radians)
        let camDist = """ + str(BLADE_LEN * 1.5) + """;  // distance

        const MIN_EL = 0.05;
        const MAX_EL = 1.5;
        const MIN_DIST = """ + str(BLADE_LEN * 0.3) + """;
        const MAX_DIST = """ + str(BLADE_LEN * 5.0) + """;

        function sendCamera() {
            log('[cam] send az=' + camAz.toFixed(3) + ' el=' + camEl.toFixed(3) + ' d=' + camDist.toFixed(1) + ' ws=' + (ws ? ws.readyState : 'null'));
            send({cmd: 'camera', azimuth: camAz, elevation: camEl, distance: camDist});
        }

        // ─── Mouse controls ───
        let dragging = false;
        let lastX = 0, lastY = 0;

        overlay.addEventListener('mousedown', (e) => {
            log('[cam] mousedown ' + e.clientX + ',' + e.clientY);
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

        overlay.addEventListener('wheel', (e) => {
            e.preventDefault();
            camDist = Math.max(MIN_DIST, Math.min(MAX_DIST, camDist * (1 + e.deltaY * 0.001)));
            sendCamera();
        }, {passive: false});

        // ─── Touch controls (on overlay to avoid iOS img conflicts) ───
        let touches = {};
        let lastPinchDist = 0;

        overlay.addEventListener('touchstart', (e) => {
            log('[cam] touchstart fingers=' + e.touches.length + ' target=' + e.target.className);
            e.preventDefault();
            e.stopPropagation();
            camHint.style.display = 'none';
            for (const t of e.changedTouches) {
                touches[t.identifier] = {x: t.clientX, y: t.clientY};
                log('[cam]   touch id=' + t.identifier + ' x=' + t.clientX.toFixed(0) + ' y=' + t.clientY.toFixed(0));
            }
            if (e.touches.length === 2) {
                const [a, b] = [e.touches[0], e.touches[1]];
                lastPinchDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
                log('[cam] pinch start dist=' + lastPinchDist.toFixed(1));
            }
        }, {passive: false});

        overlay.addEventListener('touchmove', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.touches.length === 1) {
                // Single finger: orbit
                const t = e.touches[0];
                const prev = touches[t.identifier];
                if (prev) {
                    const dx = t.clientX - prev.x;
                    const dy = t.clientY - prev.y;
                    log('[cam] touchmove orbit dx=' + dx.toFixed(1) + ' dy=' + dy.toFixed(1));
                    camAz += dx * 0.006;
                    camEl = Math.max(MIN_EL, Math.min(MAX_EL, camEl + dy * 0.006));
                    sendCamera();
                } else {
                    log('[cam] touchmove NO prev for id=' + t.identifier);
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

        overlay.addEventListener('touchend', (e) => {
            log('[cam] touchend fingers=' + e.touches.length);
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
                log('[ws] Connected');
                status.textContent = 'Connected';
                status.className = 'status connected';
                sendCamera();  // sync camera state on connect
            };
            ws.onclose = (ev) => {
                log('[ws] Closed code=' + ev.code + ' reason=' + ev.reason);
                status.textContent = 'Reconnecting...';
                status.className = 'status';
                setTimeout(connectWs, 2000);
            };
            ws.onmessage = (e) => {
                try {
                    const state = JSON.parse(e.data);
                    if (state.ack === 'camera') {
                        log('[ws] server ack cam az=' + state.az.toFixed(3) + ' el=' + state.el.toFixed(3) + ' d=' + state.d.toFixed(1));
                    }
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

async def idle_watchdog():
    """Stop the RunPod pod after IDLE_TIMEOUT_SECS of no client activity."""
    while True:
        await asyncio.sleep(60)
        idle_secs = time.time() - last_activity
        has_clients = len(clients) > 0 or len(stream_clients) > 0
        if has_clients:
            touch_activity()
            continue
        remaining = IDLE_TIMEOUT_SECS - idle_secs
        if remaining > 0:
            if remaining < 300:  # log when < 5 min left
                print(f"[idle] No clients for {idle_secs/60:.0f}m, sleeping in {remaining/60:.1f}m")
            continue
        print(f"[idle] No activity for {idle_secs/60:.0f} minutes — stopping pod")
        pod_id = os.environ.get('RUNPOD_POD_ID')
        api_key = os.environ.get('RUNPOD_API_KEY')
        try:
            if pod_id:
                subprocess.run(['runpodctl', 'stop', 'pod', pod_id], timeout=30, check=True)
            else:
                subprocess.run(['runpodctl', 'stop', 'pod', '--all'], timeout=30, check=True)
            print("[idle] runpodctl stop pod succeeded")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[idle] runpodctl failed ({e}), using API fallback")
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
                    print("[idle] API stop succeeded")
                except Exception as e:
                    print(f"[idle] API stop failed: {e}")
            else:
                print("[idle] No RUNPOD_POD_ID/RUNPOD_API_KEY — cannot stop pod")
        except Exception as e:
            print(f"[idle] Stop failed: {e}")
        break


async def start_background_tasks(app):
    app['physics_task'] = asyncio.ensure_future(physics_loop())
    app['idle_task'] = asyncio.ensure_future(idle_watchdog())


async def cleanup_background_tasks(app):
    app['physics_task'].cancel()
    app['idle_task'].cancel()


# ── entry point ───────────────────────────────────────────────────

def run_server(host='0.0.0.0', port=8765):
    app = web.Application()
    app.router.add_get('/', sandbox_handler)
    app.router.add_get('/sandbox', sandbox_handler)
    app.router.add_get('/debug_mesh', debug_mesh_handler)
    app.router.add_get('/stream', stream_handler)
    app.router.add_get('/frame', frame_handler)
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

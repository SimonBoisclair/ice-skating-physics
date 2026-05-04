import { useEffect, useRef, useCallback, useState } from 'react';
import { usePhysics } from '../context/PhysicsContext';
import iceHardness from '../utils/iceHardness';
import { SCALE } from '../constants';

/**
 * Manages the WebSocket connection to the GPU physics server.
 * Receives state updates and provides a `send` helper.
 */
export default function useWebSocket() {
  const { params, setGpu } = usePhysics();
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const send = useCallback((obj) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }, []);

  /** Send all current slider values — called once on connect. */
  const sendAllParams = useCallback(() => {
    const p = paramsRef.current;
    send({ cmd: 'lean', value: p.lean });
    send({ cmd: 'alpha', value: p.alpha });
    send({ cmd: 'pitch', value: p.pitch });
    send({ cmd: 'weight', value: p.mass });
    send({ cmd: 'ice', value: iceHardness(p.temp) });
    send({ cmd: 'set_velocity', vx: p.gvx * SCALE, vy: p.gvy * SCALE });
  }, [send]);

  useEffect(() => {
    let reconnectTimer;

    function connect() {
      // Use VITE_WS_URL env var for live backend, or default to current host
      const wsUrl = import.meta.env.VITE_WS_URL;
      let url;
      if (wsUrl) {
        url = wsUrl;
        console.log('[WebSocket] Connecting to env URL:', url);
      } else {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        url = `${proto}//${window.location.host}/ws`;
        console.log('[WebSocket] Connecting to:', url);
      }
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        sendAllParams();
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === 'state') {
            setGpu(data);
          }
        } catch {
          /* ignore non-JSON */
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [setGpu, sendAllParams]);

  return { connected, send };
}

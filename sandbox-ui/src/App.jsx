import React, { useRef, useCallback, useEffect } from 'react';
import { PhysicsProvider, usePhysics } from './context/PhysicsContext';
import useWebSocket from './hooks/useWebSocket';
import Panel from './components/Panel/Panel';
import ThreeScene from './components/Scene/ThreeScene';
import Legend from './components/Legend';
import InfoBar from './components/InfoBar';
import iceHardness from './utils/iceHardness';
import { SCALE } from './constants';

function AppInner() {
  const { params, gpu } = usePhysics();
  const { connected, send } = useWebSocket();
  const sceneRef = useRef(null);

  /** Map a param key to a WebSocket command. */
  const onParamSend = useCallback(
    (key, value) => {
      switch (key) {
        case 'lean':
          send({ cmd: 'lean', value });
          break;
        case 'alpha':
          send({ cmd: 'alpha', value });
          break;
        case 'pitch':
          send({ cmd: 'pitch', value });
          break;
        case 'mass':
          send({ cmd: 'weight', value });
          break;
        case 'temp':
          send({ cmd: 'ice', value: iceHardness(value) });
          break;
        case 'gvx':
        case 'gvy':
          // Need latest params from ref — this callback sees stale closure.
          // We'll re-read from params on next render. For now send with given value.
          send({ cmd: 'set_velocity', vx: (key === 'gvx' ? value : params.gvx) * SCALE, vy: (key === 'gvy' ? value : params.gvy) * SCALE });
          break;
        case 'gz':
          send({ cmd: 'set_L', value });
          break;
        case 'hollow':
          send({ cmd: 'hollow_radius', value });
          break;
        default:
          break;
      }
    },
    [send, params.gvx, params.gvy]
  );

  const onSetCamera = useCallback(
    (mode) => sceneRef.current?.setCamera(mode, params),
    [params]
  );

  // Update 3D scene whenever params or GPU state changes
  useEffect(() => {
    sceneRef.current?.updateScene(params, gpu);
  }, [params, gpu]);

  return (
    <>
      <Panel connected={connected} onParamSend={onParamSend} send={send} onSetCamera={onSetCamera} />
      <ThreeScene ref={sceneRef} />
      <Legend />
      <InfoBar />
    </>
  );
}

export default function App() {
  return (
    <PhysicsProvider>
      <AppInner />
    </PhysicsProvider>
  );
}

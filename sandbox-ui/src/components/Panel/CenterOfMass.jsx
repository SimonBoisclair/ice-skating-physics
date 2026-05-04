import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import SliderControl from './SliderControl';

export default function CenterOfMass({ onParamSend }) {
  const { params, setParam } = usePhysics();

  const change = (key) => (v) => {
    setParam(key, v);
    onParamSend(key, v);
  };

  return (
    <div className="sec">
      <div className="sec-t">G &mdash; CENTER OF MASS (input)</div>
      <SliderControl label="G.x (forward)" value={params.gx} min={-2} max={2} step={0.01} unit="m" onChange={change('gx')} />
      <SliderControl label="G.y (lateral)" value={params.gy} min={-1} max={1} step={0.01} unit="m" onChange={change('gy')} />
      <SliderControl label="G.z (height)" value={params.gz} min={0.1} max={1.5} step={0.01} unit="m" onChange={change('gz')} />
      <hr style={{ borderColor: '#1e2d42', margin: '4px 0' }} />
      <SliderControl label="G.vx (forward vel)" value={params.gvx} min={-15} max={15} step={0.1} unit="m/s" onChange={change('gvx')} />
      <SliderControl label="G.vy (lateral vel)" value={params.gvy} min={-5} max={5} step={0.1} unit="m/s" onChange={change('gvy')} />
      <SliderControl label="G.vz (vertical vel)" value={params.gvz} min={-5} max={5} step={0.1} unit="m/s" onChange={change('gvz')} />
      <div className="hint">Velocity shown as purple arrow (no time stepping)</div>
    </div>
  );
}

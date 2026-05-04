import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import SliderControl from './SliderControl';

export default function ContactPosition({ onParamSend }) {
  const { params, setParam } = usePhysics();

  const change = (key) => (v) => {
    setParam(key, v);
    onParamSend(key, v);
  };

  return (
    <div className="sec">
      <div className="sec-t">P &mdash; CONTACT CENTER (input)</div>
      <SliderControl label="P.x (forward)" value={params.px} min={-2} max={2} step={0.01} unit="m" onChange={change('px')} />
      <SliderControl label="P.y (lateral)" value={params.py} min={-2} max={2} step={0.01} unit="m" onChange={change('py')} />
      <div className="hint">Center of penetration polygon on ice surface</div>
    </div>
  );
}

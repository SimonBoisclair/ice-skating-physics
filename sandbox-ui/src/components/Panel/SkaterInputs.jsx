import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import SliderControl from './SliderControl';

export default function SkaterInputs({ onParamSend }) {
  const { params, setParam } = usePhysics();

  const change = (key) => (v) => {
    setParam(key, v);
    onParamSend(key, v);
  };

  return (
    <div className="sec">
      <div className="sec-t">SKATER (input)</div>
      <SliderControl label="Mass" value={params.mass} min={30} max={120} step={1} unit="kg" onChange={change('mass')} />
    </div>
  );
}

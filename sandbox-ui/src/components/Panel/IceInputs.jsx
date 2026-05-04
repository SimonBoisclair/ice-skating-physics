import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import SliderControl from './SliderControl';

export default function IceInputs({ onParamSend }) {
  const { params, setParam } = usePhysics();

  const change = (key) => (v) => {
    setParam(key, v);
    onParamSend(key, v);
  };

  return (
    <div className="sec">
      <div className="sec-t">ICE (input)</div>
      <SliderControl
        label="Temperature"
        value={params.temp}
        min={-25} max={-1} step={0.5} unit={'\u00B0C'}
        hint={'-1°C=warm/soft · -5°C=rink · -20°C=outdoor cold'}
        onChange={change('temp')}
      />
    </div>
  );
}

import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import SliderControl from './SliderControl';

export default function BladeInputs({ onParamSend }) {
  const { params, setParam } = usePhysics();

  const change = (key) => (v) => {
    setParam(key, v);
    onParamSend(key, v);
  };

  return (
    <div className="sec">
      <div className="sec-t">BLADE (input)</div>
      <SliderControl
        label={'\u03B1 (foot opening)'}
        value={params.alpha}
        min={0} max={90} step={1} unit={'\u00B0'}
        hint={'0°=glide · 45°=push · 90°=hockey stop'}
        onChange={change('alpha')}
      />
      <SliderControl
        label="Lean angle"
        value={params.lean}
        min={0} max={60} step={1} unit={'\u00B0'}
        hint={'0°=flat · 45°=deep edge · >55°=boot contact'}
        onChange={change('lean')}
      />
      <SliderControl
        label="Pitch (rocker)"
        value={params.pitch}
        min={-1} max={1} step={0.05}
        hint={'-1=heel · 0=center · +1=toe'}
        onChange={change('pitch')}
      />
    </div>
  );
}

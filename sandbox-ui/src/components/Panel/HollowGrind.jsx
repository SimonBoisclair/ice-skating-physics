import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import SliderControl from './SliderControl';

export default function HollowGrind({ onParamSend }) {
  const { params, setParam } = usePhysics();

  const change = (v) => {
    setParam('hollow', v);
    onParamSend('hollow', v);
  };

  return (
    <div className="sec">
      <div className="sec-t">HOLLOW GRIND</div>
      <SliderControl label="Hollow (mm)" value={params.hollow} min={5} max={50} step={0.5} unit="mm" onChange={change} />
    </div>
  );
}

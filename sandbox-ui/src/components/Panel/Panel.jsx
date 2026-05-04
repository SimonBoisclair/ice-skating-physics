import React from 'react';
import CenterOfMass from './CenterOfMass';
import BladeInputs from './BladeInputs';
import ContactPosition from './ContactPosition';
import SkaterInputs from './SkaterInputs';
import IceInputs from './IceInputs';
import GpuReadouts from './GpuReadouts';
import CameraButtons from './CameraButtons';
import BladeProfileLegend from './BladeProfileLegend';
import HollowGrind from './HollowGrind';

export default function Panel({ connected, onParamSend, send, onSetCamera }) {
  return (
    <div id="panel">
      <h1>ICE SKATING PHYSICS</h1>
      <div className="sub">Connected to GPU mesh collision engine</div>

      <div
        id="wsStatus"
        style={{
          background: connected ? '#2ecc40' : '#e74c3c',
          color: connected ? '#000' : '#fff',
          padding: '3px 8px',
          borderRadius: '3px',
          fontSize: '10px',
          fontWeight: 'bold',
          margin: '4px 0',
        }}
      >
        {connected ? 'GPU Engine Connected' : 'Disconnected'}
      </div>

      <CenterOfMass onParamSend={onParamSend} />
      <BladeInputs onParamSend={onParamSend} />
      <ContactPosition onParamSend={onParamSend} />
      <SkaterInputs onParamSend={onParamSend} />
      <IceInputs onParamSend={onParamSend} />
      <GpuReadouts send={send} />
      <CameraButtons onSetCamera={onSetCamera} />
      <BladeProfileLegend />
      <HollowGrind onParamSend={onParamSend} />
    </div>
  );
}

import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';

export default function PenetrationControls({ send }) {
  const { state } = usePhysics();
  
  const physicsPaused = state?.physics_paused ?? true;
  const bladeAtSurface = state?.blade_at_surface ?? false;

  const handleResetBlade = () => {
    send({ cmd: 'reset_blade_position' });
  };

  const handleStartPenetration = () => {
    send({ cmd: 'start_penetration' });
  };

  return (
    <fieldset>
      <legend>Penetration Calculation</legend>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <button
          onClick={handleResetBlade}
          style={{
            padding: '8px 12px',
            background: '#3498db',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          Reset Blade to Surface
        </button>

        {bladeAtSurface && physicsPaused && (
          <button
            onClick={handleStartPenetration}
            style={{
              padding: '8px 12px',
              background: '#2ecc40',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            Start Penetration Calculation
          </button>
        )}

        <div style={{ fontSize: '10px', color: '#888' }}>
          {physicsPaused ? (
            bladeAtSurface ? 
              'Blade at surface. Click to start physics.' : 
              'Physics paused. Reset blade first.'
          ) : (
            'Physics running...'
          )}
        </div>
      </div>
    </fieldset>
  );
}

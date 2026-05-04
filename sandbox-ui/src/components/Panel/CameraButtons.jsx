import React, { useState } from 'react';

const MODES = ['persp', 'side', 'front', 'top', 'blade'];

export default function CameraButtons({ onSetCamera }) {
  const [active, setActive] = useState('side');

  const handleClick = (mode) => {
    setActive(mode);
    onSetCamera(mode);
  };

  return (
    <div className="sec">
      <div className="sec-t">CAMERA</div>
      <div className="btn-row">
        {MODES.map((m) => (
          <button key={m} className={active === m ? 'active' : ''} onClick={() => handleClick(m)}>
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
}

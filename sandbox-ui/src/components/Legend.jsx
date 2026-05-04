import React from 'react';

const ITEMS = [
  { color: '#0074D9', label: 'L line (G\u2192P) / F_body' },
  { color: '#2ecc40', label: 'F_gravity (mg \u2193)' },
  { color: '#B10DC9', label: 'Velocity vector' },
  { color: '#FF851B', label: 'Penetration polygon' },
  { color: '#ff4136', label: 'R_across (groove resist)' },
  { color: '#2ecc40', label: 'R_along (groove fric)' },
  { color: '#FFDC00', label: '\u03B8 arc / arc path' },
];

export default function Legend() {
  return (
    <div id="legend">
      {ITEMS.map((item) => (
        <div className="lr" key={item.label}>
          <span className="lc" style={{ background: item.color }} />
          {item.label}
        </div>
      ))}
    </div>
  );
}

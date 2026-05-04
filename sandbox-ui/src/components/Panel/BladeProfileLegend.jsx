import React from 'react';

const ZONES = [
  { color: '#ff4444', label: "Heel/Toe Tips (0.15')" },
  { color: '#9b59b6', label: "Zone 4 (15') \u2190 heel" },
  { color: '#3498db', label: "Zone 3 (12')" },
  { color: '#2ecc40', label: "Zone 2 (9') \u2190 center" },
  { color: '#f1c40f', label: "Zone 1 (6') \u2190 toe" },
];

export default function BladeProfileLegend() {
  return (
    <div className="sec">
      <div className="sec-t">BLADE PROFILE (QUAD 1)</div>
      {ZONES.map((z) => (
        <div className="zone-i" key={z.label}>
          <span className="zone-d" style={{ background: z.color }} />
          {z.label}
        </div>
      ))}
      <div className="hint">280mm blade &middot; 5/8&quot; hollow</div>
    </div>
  );
}

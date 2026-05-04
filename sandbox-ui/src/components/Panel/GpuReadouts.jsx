import React from 'react';
import { usePhysics } from '../../context/PhysicsContext';
import { G_ACC } from '../../constants';

function Row({ label, value, unit, color }) {
  return (
    <div className="pr">
      <span className="pl">{label}</span>
      <span>
        <span className={`pv ${color || ''}`}>{value}</span>
        {unit && <span className="pu">{unit}</span>}
      </span>
    </div>
  );
}

export default function GpuReadouts({ send }) {
  const { params, gpu: g } = usePhysics();

  const fmt = (v, d = 1) => (v !== undefined ? Number(v).toFixed(d) : '---');

  // Derived formula display
  let formulaText = '---';
  if (g.pen_analytical_mm !== undefined && g.ice_hardness_mpa !== undefined) {
    const Fn = (params.mass * G_ACC * Math.cos((params.lean * Math.PI) / 180)).toFixed(0);
    const H = g.ice_hardness_mpa;
    const Lc = (g.contact_length_mm || 0).toFixed(1);
    const w = (g.contact_width_mm || 0).toFixed(2);
    const resist = (
      H * 1e6 * ((g.contact_length_mm || 1) / 1000) * ((g.contact_width_mm || 0.1) / 1000)
    ).toFixed(0);
    formulaText = `${H}MPa\u00D7${Lc}mm\u00D7${w}mm=${resist}N vs F=${Fn}N`;
  }

  return (
    <div className="sec">
      <div className="sec-t">GPU-COMPUTED (real mesh collision)</div>

      <Row label="Collision mode" value={g.collision_mode ? g.collision_mode.toUpperCase() : '---'} color="blu" />
      <Row label="Hollow radius" value={fmt(g.hollow_radius_mm, 1)} unit="mm" />
      <hr style={{ borderColor: '#1e2d42', margin: '3px 0' }} />

      <Row label="Speed" value={fmt(g.speed, 3)} unit="m/s" color="grn" />
      <Row label="v_along (blade)" value={fmt(g.v_along, 3)} unit="m/s" />
      <Row label="v_perp (across)" value={fmt(g.v_perp, 3)} unit="m/s" />
      <hr style={{ borderColor: '#1e2d42', margin: '3px 0' }} />

      <Row label="F_gravity" value={fmt(g.Fg, 1)} unit={'N ↓'} color="grn" />
      <Row label="F_lateral (GPU)" value={fmt(g.f_lateral, 0)} unit="N" color="red" />
      <Row label="F_along (GPU)" value={fmt(g.f_along, 0)} unit="N" color="grn" />
      <Row label="Pen. max (GPU)" value={fmt(g.pen, 3)} unit="mm" color="org" />
      <Row label="Pen. avg (GPU)" value={fmt(g.pen_avg_mm, 3)} unit="mm" color="org" />
      <Row label="Contact particles" value={g.pen_contact_count ?? '---'} />
      <Row label="Contact area (particles)" value={fmt(g.pen_contact_area_mm2, 1)} unit={'mm²'} />
      <hr style={{ borderColor: '#1e2d42', margin: '3px 0' }} />

      <Row label="Contact length (STL)" value={fmt(g.contact_length_mm, 1)} unit="mm" color="cyn" />
      <Row label="Contact width (STL)" value={fmt(g.contact_width_mm, 2)} unit="mm" color="cyn" />
      <Row label="Contact area (STL)" value={fmt(g.contact_area_geom_mm2, 1)} unit={'mm²'} color="cyn" />
      <Row label="Pen. analytical" value={fmt(g.pen_analytical_mm, 3)} unit="mm" color="pur" />

      <div className="pr" style={{ fontSize: '8px', color: '#4a6a8a' }}>
        <span className="pl" style={{ fontSize: '8px' }}>
          H&times;L<sub>c</sub>(d)&times;w(d)=F<sub>n</sub>
        </span>
        <span className="pv" style={{ fontSize: '8px', color: '#4a6a8a' }}>{formulaText}</span>
      </div>

      <Row label="F_reaction (z)" value={fmt(g.blade_reaction_z, 1)} unit="N" />
      <Row label="F_normal (target)" value={fmt(g.F_normal, 1)} unit="N" />
      <hr style={{ borderColor: '#1e2d42', margin: '3px 0' }} />

      <Row label="Lean (actual)" value={fmt(g.theta_actual, 1)} unit={'\u00B0'} color="ylw" />
      <Row label={'\u03B1 (foot opening)'} value={fmt(g.alpha, 1)} unit={'\u00B0'} />
      <Row label="Active zone" value={g.zone_name || '---'} />
      <Row label="Turn radius R" value={g.R !== undefined && isFinite(g.R) ? Number(g.R).toFixed(2) : '\u221E'} unit="m" />
      <Row label={'\u03B8_balance (required)'} value={fmt(g.theta_balance, 1)} unit={'\u00B0'} color="ylw" />
      <Row label={'L (G→P dist)'} value={fmt(g.L, 2)} unit="m" color="blu" />
      <hr style={{ borderColor: '#1e2d42', margin: '3px 0' }} />

      <Row label="SPS" value={fmt(g.sps, 0)} />
      <Row label="Frame" value={g.frame ?? '---'} />
      <Row label="Particles" value={g.n_ice ?? '---'} />

      <div className="btn-row">
        <button onClick={() => send({ cmd: 'toggle_mesh' })}>Toggle Mesh/Box</button>
        <button onClick={() => send({ cmd: 'reset' })}>Reset Ice</button>
        <button onClick={() => send({ cmd: 'push', fx: 0, fy: 0, force: 2 })}>Push</button>
      </div>
    </div>
  );
}

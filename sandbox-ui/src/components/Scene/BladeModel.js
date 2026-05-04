import * as THREE from 'three';
import computeProfileCurve from '../../utils/bladeProfile';
import {
  BLADE_LEN,
  BLADE_T,
  BLADE_H,
  BLADE_HALF,
  HOLLOW_R,
  PROFILE_ZONES,
  PROFILE_RADII_FT,
  ZONE_COLORS_HEX,
} from '../../constants';

/**
 * Build the full blade Three.js Group:
 *  - colour-coded bottom (hollow grind profile)
 *  - top surface
 *  - side walls + end caps
 *  - orange penetration clone (below-ice clipping)
 *  - toe arrow
 *  - TUUK holder
 *
 * Returns { group, profile } where profile is the computed rocker curve.
 */
export default function buildBlade(iceClipPlane) {
  const group = new THREE.Group();
  const profile = { zones: PROFILE_ZONES, radii: PROFILE_RADII_FT };
  const halfLen = BLADE_HALF;
  const halfT = BLADE_T / 2;
  const curve = computeProfileCurve(BLADE_LEN, profile);

  function byi(x) {
    const t = (x + halfLen) / BLADE_LEN;
    const fi = t * curve.n;
    const i0 = Math.max(0, Math.min(curve.n - 1, Math.floor(fi)));
    const i1 = Math.min(curve.n, i0 + 1);
    const f = fi - i0;
    return curve.y[i0] * (1 - f) + curve.y[i1] * f;
  }

  function gz(x) {
    const t = (x + halfLen) / BLADE_LEN;
    const i = Math.max(0, Math.min(curve.n, Math.round(t * curve.n)));
    return curve.zone[i];
  }

  function hd(z) {
    const r2 = HOLLOW_R * HOLLOW_R;
    const ht2 = halfT * halfT;
    if (ht2 >= r2) return 0;
    return Math.sqrt(Math.max(r2 - z * z, 0)) - Math.sqrt(r2 - ht2);
  }

  const nL = 120;
  const nT = 20;
  const pos = [];
  const nor = [];
  const idx = [];
  const col = [];

  // Bottom
  for (let i = 0; i <= nL; i++) {
    const x = -halfLen + (i / nL) * BLADE_LEN;
    const yb = byi(x);
    const z = gz(x);
    const c = new THREE.Color(ZONE_COLORS_HEX[z]);
    for (let j = 0; j <= nT; j++) {
      const zz = -halfT + (j / nT) * BLADE_T;
      pos.push(x, yb + hd(zz), zz);
      nor.push(0, -1, 0);
      col.push(c.r, c.g, c.b);
    }
  }
  for (let i = 0; i < nL; i++) {
    for (let j = 0; j < nT; j++) {
      const a = i * (nT + 1) + j;
      const b = a + 1;
      const c = (i + 1) * (nT + 1) + j;
      const d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }

  // Top
  const to = pos.length / 3;
  for (let i = 0; i <= nL; i++) {
    const x = -halfLen + (i / nL) * BLADE_LEN;
    const yb = byi(x);
    const z = gz(x);
    const c = new THREE.Color(ZONE_COLORS_HEX[z]);
    for (let j = 0; j <= nT; j++) {
      pos.push(x, yb + BLADE_H, -halfT + (j / nT) * BLADE_T);
      nor.push(0, 1, 0);
      col.push(c.r, c.g, c.b);
    }
  }
  for (let i = 0; i < nL; i++) {
    for (let j = 0; j < nT; j++) {
      const a = to + i * (nT + 1) + j;
      const b = a + 1;
      const c = to + (i + 1) * (nT + 1) + j;
      const d = c + 1;
      idx.push(a, b, c, b, d, c);
    }
  }

  // Sides
  function addSide(zv, nd) {
    const o = pos.length / 3;
    for (let i = 0; i <= nL; i++) {
      const x = -halfLen + (i / nL) * BLADE_LEN;
      const yb = byi(x) + hd(zv);
      const yt = byi(x) + BLADE_H;
      const z = gz(x);
      const c = new THREE.Color(ZONE_COLORS_HEX[z]);
      pos.push(x, yb, zv, x, yt, zv);
      nor.push(0, 0, nd, 0, 0, nd);
      col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
    for (let i = 0; i < nL; i++) {
      const a = o + i * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      if (nd > 0) idx.push(a, c, b, b, c, d);
      else idx.push(a, b, c, b, d, c);
    }
  }
  addSide(halfT, 1);
  addSide(-halfT, -1);

  // End caps
  function addCap(xp, nx) {
    const o = pos.length / 3;
    const yb1 = byi(xp) + hd(-halfT);
    const yb2 = byi(xp) + hd(halfT);
    const yt = byi(xp) + BLADE_H;
    const z = gz(xp);
    const c = new THREE.Color(ZONE_COLORS_HEX[z]);
    pos.push(xp, yb1, -halfT, xp, yb2, halfT, xp, yt, -halfT, xp, yt, halfT);
    for (let k = 0; k < 4; k++) {
      nor.push(nx, 0, 0);
      col.push(c.r, c.g, c.b);
    }
    if (nx > 0) idx.push(o, o + 1, o + 2, o + 1, o + 3, o + 2);
    else idx.push(o, o + 2, o + 1, o + 1, o + 2, o + 3);
  }
  addCap(halfLen, 1);
  addCap(-halfLen, -1);

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geom.setIndex(idx);
  geom.computeVertexNormals();

  // Normal blade — clipped above ice
  const aboveIcePlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const bladeMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.15,
    metalness: 0.85,
    side: THREE.DoubleSide,
    clippingPlanes: [aboveIcePlane],
  });
  group.add(new THREE.Mesh(geom, bladeMat));

  // Orange penetration clone — below ice
  const penBladeMat = new THREE.MeshStandardMaterial({
    color: 0xff851b,
    emissive: 0xff851b,
    emissiveIntensity: 0.6,
    roughness: 0.3,
    metalness: 0.4,
    side: THREE.DoubleSide,
    clippingPlanes: [iceClipPlane],
  });
  group.add(new THREE.Mesh(geom, penBladeMat));

  // Toe arrow
  const ar = new THREE.Mesh(
    new THREE.ConeGeometry(0.005, 0.015, 8),
    new THREE.MeshStandardMaterial({ color: 0xff0000 })
  );
  ar.rotation.z = -Math.PI / 2;
  ar.position.set(halfLen + 0.01, byi(halfLen) + BLADE_H * 0.7, 0);
  group.add(ar);

  // TUUK holder
  {
    const hH = 0.048;
    const hT2 = 0.011;
    const hHT = hT2 / 2;
    const hf = PROFILE_ZONES[0];
    const tf = PROFILE_ZONES[5];
    const hsx = -halfLen + hf * BLADE_LEN * 0.35;
    const hex = halfLen - tf * BLADE_LEN * 0.35;
    const hL = hex - hsx;
    const hby = byi(0) + BLADE_H;
    const hm = new THREE.MeshStandardMaterial({ color: 0xf0f0f0, roughness: 0.3, metalness: 0.03 });

    function mkHS() {
      const L = hL;
      const H = hH;
      const s = new THREE.Shape();
      s.moveTo(0, 0);
      s.lineTo(L, 0);
      s.quadraticCurveTo(L * 1.02, H * 0.15, L * 1.01, H * 0.4);
      s.quadraticCurveTo(L * 1.005, H * 0.75, L * 0.98, H);
      s.quadraticCurveTo(L * 0.75, H * 1.02, L * 0.5, H * 1.01);
      s.quadraticCurveTo(L * 0.25, H * 1.005, L * 0.02, H);
      s.quadraticCurveTo(-L * 0.005, H * 0.75, -L * 0.01, H * 0.4);
      s.quadraticCurveTo(-L * 0.005, H * 0.15, 0, 0);

      [
        [0.06, 0.19, 0.12, 0.86],
        [0.25, 0.49, 0.1, 0.88],
        [0.55, 0.91, 0.1, 0.88],
      ].forEach(([l, r, b, t]) => {
        const w = new THREE.Path();
        const wL = L * l;
        const wR = L * r;
        const wB = H * b;
        const wT = H * t;
        w.moveTo(wL, wB);
        w.quadraticCurveTo((wL + wR) / 2, wB - H * 0.02, wR, wB);
        w.quadraticCurveTo(wR + L * 0.015, (wB + wT) / 2, wR, wT);
        w.quadraticCurveTo((wL + wR) / 2, wT + H * 0.02, wL, wT);
        w.quadraticCurveTo(wL - L * 0.015, (wB + wT) / 2, wL, wB);
        s.holes.push(w);
      });
      return s;
    }

    const hg = new THREE.ExtrudeGeometry(mkHS(), {
      depth: hT2,
      bevelEnabled: true,
      bevelThickness: 0.0004,
      bevelSize: 0.0004,
      bevelSegments: 3,
    });
    const hmesh = new THREE.Mesh(hg, hm);
    hmesh.position.set(hsx, hby, -hHT);
    hmesh.castShadow = true;
    group.add(hmesh);

    const rm = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.25, metalness: 0.85 });
    [0.1, 0.22, 0.37, 0.52, 0.7, 0.85, 0.95].forEach((f) => {
      const rx = hsx + hL * f;
      const rg = new THREE.CylinderGeometry(0.0009, 0.0009, hT2 + 0.002, 8);
      const r = new THREE.Mesh(rg, rm);
      r.rotation.x = Math.PI / 2;
      r.position.set(rx, hby + hH - 0.003, 0);
      group.add(r);
    });
  }

  return { group, profile: curve };
}

import { BLADE_H } from '../constants';

/**
 * Compute the rocker profile curve for the blade.
 * Returns { y, slope, zone, n, halfLen } arrays describing the
 * height along the blade length for rendering.
 */
export default function computeProfileCurve(bladeLen, profile) {
  const n = 400;
  const dx = bladeLen / n;
  const radiiM = profile.radii.map((r) => r * 0.3048);
  const zoneBounds = [];
  let cumFrac = 0;
  for (const f of profile.zones) {
    zoneBounds.push(cumFrac);
    cumFrac += f;
  }
  zoneBounds.push(1.0);
  const halfLen = bladeLen / 2;

  function getR(x) {
    const t = (x + halfLen) / bladeLen;
    for (let z = 1; z < profile.radii.length - 1; z++) {
      if (t <= zoneBounds[z + 1] + 0.001) return radiiM[z];
    }
    return radiiM[profile.radii.length - 2];
  }

  function getZ(x) {
    const t = (x + halfLen) / bladeLen;
    for (let z = 0; z < profile.radii.length; z++) {
      if (t <= zoneBounds[z + 1] + 0.001) return z;
    }
    return profile.radii.length - 1;
  }

  const y = new Float64Array(n + 1);
  const slope = new Float64Array(n + 1);
  const zone = new Int32Array(n + 1);

  const ci = Math.round(n / 2);
  const hbi = Math.round(zoneBounds[1] * n);
  const tbi = Math.round(zoneBounds[profile.radii.length - 1] * n);

  // Forward from centre to toe boundary
  for (let i = ci; i < tbi && i < n; i++) {
    const x = -halfLen + (i / n) * bladeLen;
    const R = getR(x);
    slope[i + 1] = slope[i] + (1 / R) * dx;
    y[i + 1] = y[i] + slope[i] * dx + 0.5 * (1 / R) * dx * dx;
    zone[i] = getZ(x);
  }

  // Backward from centre to heel boundary
  for (let i = ci; i > hbi && i > 0; i--) {
    const x = -halfLen + (i / n) * bladeLen;
    const R = getR(x);
    slope[i - 1] = slope[i] - (1 / R) * dx;
    y[i - 1] = y[i] - slope[i] * dx + 0.5 * (1 / R) * dx * dx;
    zone[i - 1] = getZ(-halfLen + ((i - 1) / n) * bladeLen);
  }

  if (tbi <= n) zone[tbi] = getZ(-halfLen + (tbi / n) * bladeLen);

  // Tip curves
  const tipH = BLADE_H * 0.85;
  const tipP = 3;
  {
    const sy = y[tbi];
    const ss = slope[tbi];
    const tl = (n - tbi) * dx;
    for (let i = tbi + 1; i <= n; i++) {
      const lx = (i - tbi) * dx;
      const t = lx / tl;
      y[i] = sy + ss * lx + tipH * Math.pow(t, tipP);
      slope[i] = ss + (tipH * tipP * Math.pow(t, tipP - 1)) / tl;
      zone[i] = profile.radii.length - 1;
    }
  }
  {
    const sy = y[hbi];
    const ss = slope[hbi];
    const tl = hbi * dx;
    for (let i = hbi - 1; i >= 0; i--) {
      const lx = (hbi - i) * dx;
      const t = lx / tl;
      y[i] = sy - ss * lx + tipH * Math.pow(t, tipP);
      slope[i] = ss - (tipH * tipP * Math.pow(t, tipP - 1)) / tl;
      zone[i] = 0;
    }
  }

  let minY = Infinity;
  for (let i = 0; i <= n; i++) minY = Math.min(minY, y[i]);
  for (let i = 0; i <= n; i++) y[i] -= minY;

  return { y, slope, zone, n, halfLen };
}

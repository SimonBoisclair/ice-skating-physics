/**
 * Empirical ice hardness (MPa) from surface temperature (°C).
 * Clamps to [1, 20] MPa.
 */
export default function iceHardness(tempC) {
  return Math.max(1, Math.min(20, 0.65 * Math.abs(tempC) + 1.5));
}

/** Shared physics and rendering constants (must match warp_server.py) */

export const G_ACC = 9.81;
export const SCALE = 50;

// Blade geometry (metres)
export const BLADE_LEN = 0.28;
export const BLADE_T = 0.003;
export const BLADE_H = 0.03;
export const HOLLOW_R = (5 / 8) * 0.0254; // 5/8" in metres
export const BLADE_HALF = BLADE_LEN / 2;

// Friction
export const MU_ALONG = 0.005;

// Rocker profile
export const PROFILE_ZONES = [0.08, 0.18, 0.24, 0.24, 0.18, 0.08];
export const PROFILE_RADII_FT = [0.15, 15, 12, 9, 6, 0.15];
export const ZONE_NAMES = [
  'Heel Tip',
  "Zone 4 (15')",
  "Zone 3 (12')",
  "Zone 2 (9')",
  "Zone 1 (6')",
  'Toe Tip',
];
export const ZONE_COLORS_HEX = [0xff4444, 0x9b59b6, 0x3498db, 0x2ecc40, 0xf1c40f, 0xff4444];
export const ZONE_RADII_M = PROFILE_RADII_FT.map((r) => r * 0.3048);
export const MAX_PITCH_RAD = Math.asin(0.05 / BLADE_LEN);

// Panel width (pixels)
export const PANEL_WIDTH = 310;

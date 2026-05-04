import React from 'react';

/**
 * Reusable labelled slider + number input.
 *
 * Props:
 *   label   – display name
 *   value   – current numeric value
 *   min, max, step – range attributes
 *   unit    – suffix shown after the number input (e.g. "kg", "°")
 *   hint    – optional small helper text below the slider
 *   onChange(value: number) – called on every input change
 */
export default function SliderControl({ label, value, min, max, step, unit, hint, onChange }) {
  const handleSlider = (e) => onChange(parseFloat(e.target.value));
  const handleNumber = (e) => onChange(parseFloat(e.target.value));

  return (
    <div className="cr">
      <div className="ch">
        <span className="cn">{label}</span>
        <span className="cv-wrap">
          <input
            type="number"
            className="ni"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={handleNumber}
          />
          {unit && <span className="pu">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleSlider}
      />
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

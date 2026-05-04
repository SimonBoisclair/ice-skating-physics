import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import buildBlade from './BladeModel';
import { BLADE_HALF, MAX_PITCH_RAD, PANEL_WIDTH, G_ACC } from '../../constants';

/**
 * Imperative Three.js scene.
 * Exposes `updateScene(params, gpu)` and `setCamera(mode)` via ref.
 */
const ThreeScene = forwardRef(function ThreeScene(_, ref) {
  const mountRef = useRef(null);
  const internalsRef = useRef(null);

  useEffect(() => {
    const PW = PANEL_WIDTH;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a2a3e);

    const camera = new THREE.PerspectiveCamera(50, (innerWidth - PW) / innerHeight, 0.005, 200);
    camera.position.set(0.8, -1.2, 0.7);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(innerWidth - PW, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.localClippingEnabled = true;
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.3);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const sun = new THREE.DirectionalLight(0xffffff, 0.85);
    sun.position.set(3, -4, 6);
    sun.castShadow = true;
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x8899bb, 0.25);
    fill.position.set(-2, 3, 2);
    scene.add(fill);

    // Ice surface
    const iceMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(30, 30),
      new THREE.MeshStandardMaterial({
        color: 0xc8ddf0,
        roughness: 0.03,
        metalness: 0.08,
        transparent: true,
        opacity: 0.92,
      })
    );
    iceMesh.receiveShadow = true;
    scene.add(iceMesh);

    // Grid
    const grid = new THREE.GridHelper(10, 50, 0x4466aa, 0x334477);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.001;
    scene.add(grid);

    // Clip planes
    const iceClipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);

    // Blade assembly hierarchy
    const { group: bladeGroup } = buildBlade(iceClipPlane);
    const yUpGroup = new THREE.Group();
    yUpGroup.add(bladeGroup);
    yUpGroup.rotation.x = Math.PI / 2;

    const pitchGroup = new THREE.Group();
    pitchGroup.add(yUpGroup);

    const leanGroup = new THREE.Group();
    leanGroup.add(pitchGroup);

    const bladeAssembly = new THREE.Group();
    bladeAssembly.add(leanGroup);
    scene.add(bladeAssembly);

    const LEAN_AXIS = new THREE.Vector3(1, 0, 0);
    const PITCH_AXIS = new THREE.Vector3(0, 1, 0);

    // ── Helper: create canvas text sprite ──
    function mkLbl(text, color) {
      const cv = document.createElement('canvas');
      cv.width = 128;
      cv.height = 64;
      const ctx = cv.getContext('2d');
      ctx.font = 'bold 36px Arial';
      ctx.fillStyle = color || '#fff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 64, 32);
      const sp = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true })
      );
      sp.scale.set(0.07, 0.035, 1);
      return sp;
    }

    // ── Markers ──
    const pMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.008, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0xff4136, emissive: 0xff4136, emissiveIntensity: 0.3 })
    );
    scene.add(pMarker);
    const pLbl = mkLbl('P', '#ff4136');
    scene.add(pLbl);

    const gMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 20, 20),
      new THREE.MeshStandardMaterial({ color: 0x0074d9, emissive: 0x0074d9, emissiveIntensity: 0.3 })
    );
    scene.add(gMarker);
    const gLbl = mkLbl('G', '#0074D9');
    scene.add(gLbl);

    // ── Line helpers ──
    function mkLine(color, dashed) {
      const buf = new Float32Array(6);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
      const mat = dashed
        ? new THREE.LineDashedMaterial({ color, dashSize: 0.01, gapSize: 0.008 })
        : new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      scene.add(line);
      return { buf, geo, line };
    }
    function mkArcLine(n, color, dashed) {
      const buf = new Float32Array(n * 3);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
      const mat = dashed
        ? new THREE.LineDashedMaterial({ color, dashSize: 0.03, gapSize: 0.015 })
        : new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      scene.add(line);
      return { buf, geo, line };
    }

    const lLine = mkLine(0x0074d9);
    const fgLine = mkLine(0x2ecc40);
    const fgHead = new THREE.Mesh(
      new THREE.ConeGeometry(0.01, 0.025, 8),
      new THREE.MeshStandardMaterial({ color: 0x2ecc40 })
    );
    scene.add(fgHead);
    const fgLbl = mkLbl('Fg=mg', '#2ecc40');
    scene.add(fgLbl);

    const velLine = mkLine(0xb10dc9);
    const velHead = new THREE.Mesh(
      new THREE.ConeGeometry(0.008, 0.02, 8),
      new THREE.MeshStandardMaterial({ color: 0xb10dc9 })
    );
    scene.add(velHead);
    const velLbl = mkLbl('v', '#B10DC9');
    scene.add(velLbl);

    const fbLine = mkLine(0x0074d9, true);
    const fbLbl = mkLbl('F_body', '#0074D9');
    scene.add(fbLbl);

    const raLine = mkLine(0x2ecc40);
    const raLbl = mkLbl('R_along', '#2ecc40');
    scene.add(raLbl);

    const rcLine = mkLine(0xff4136);
    const rcLbl = mkLbl('R_across', '#ff4136');
    scene.add(rcLbl);

    const thArc = mkArcLine(33, 0xffdc00);
    const thLbl = mkLbl('\u03B8', '#FFDC00');
    scene.add(thLbl);

    const alArc = mkArcLine(33, 0xff851b);
    const alLbl = mkLbl('\u03B1', '#FF851B');
    scene.add(alLbl);

    const tvLine = mkLine(0xb10dc9);
    tvLine.line.material.transparent = true;
    tvLine.line.material.opacity = 0.4;

    const arcPath = mkArcLine(201, 0xffdc00, true);
    const rLine = mkLine(0x888888, true);
    rLine.line.material.transparent = true;
    rLine.line.material.opacity = 0.4;
    const rLbl = mkLbl('R', '#888');
    scene.add(rLbl);

    // ── Render loop ──
    let frameId;
    function animate() {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // ── Resize ──
    function onResize() {
      camera.aspect = (innerWidth - PW) / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth - PW, innerHeight);
    }
    window.addEventListener('resize', onResize);

    // Store internals for imperative updates
    internalsRef.current = {
      scene,
      camera,
      controls,
      renderer,
      bladeAssembly,
      leanGroup,
      pitchGroup,
      LEAN_AXIS,
      PITCH_AXIS,
      pMarker,
      pLbl,
      gMarker,
      gLbl,
      lLine,
      fgLine,
      fgHead,
      fgLbl,
      velLine,
      velHead,
      velLbl,
      fbLine,
      fbLbl,
      raLine,
      raLbl,
      rcLine,
      rcLbl,
      thArc,
      thLbl,
      alArc,
      alLbl,
      tvLine,
      arcPath,
      rLine,
      rLbl,
    };

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  useImperativeHandle(ref, () => ({
    /** Update all 3D objects from current params + GPU state. */
    updateScene(params, gpu) {
      const I = internalsRef.current;
      if (!I) return;
      const S = params;
      const g = gpu || {};

      const alphaRad = (S.alpha * Math.PI) / 180;
      const leanRad = (S.lean * Math.PI) / 180;
      const pitchAngle = S.pitch * MAX_PITCH_RAD;
      const zCorr = BLADE_HALF * Math.abs(Math.sin(pitchAngle));
      const contactCenterX = S.pitch * BLADE_HALF * 0.6;

      // Derived scene data
      const dx = S.gx - S.px;
      const dy = S.gy - S.py;
      const dz = S.gz;
      const dh = Math.sqrt(dx * dx + dy * dy);
      const L = Math.sqrt(dh * dh + dz * dz);
      const theta = Math.atan2(dh, dz);
      const thetaDeg = (theta * 180) / Math.PI;

      const speed = g.speed !== undefined ? g.speed : Math.sqrt(S.gvx * S.gvx + S.gvy * S.gvy);
      const Fg = g.Fg !== undefined ? g.Fg : S.mass * G_ACC;
      const f_lateral = g.f_lateral !== undefined ? Math.abs(g.f_lateral) : 0;
      const f_along = g.f_along !== undefined ? Math.abs(g.f_along) : 0;
      const penDepth = g.pen !== undefined ? g.pen / 1000 : 0;
      const R_turn = g.R !== undefined ? g.R : Infinity;
      const vel = g.vel || [S.gvx, S.gvy, 0];

      const bladeDir = [Math.cos(alphaRad), Math.sin(alphaRad)];
      const v_along = vel[0] * bladeDir[0] + vel[1] * bladeDir[1];
      const v_across = -vel[0] * bladeDir[1] + vel[1] * bladeDir[0];

      // ── Blade assembly ──
      const ca = Math.cos(alphaRad);
      const sa = Math.sin(alphaRad);
      I.bladeAssembly.position.set(S.px - contactCenterX * ca, S.py - contactCenterX * sa, zCorr - penDepth);
      I.bladeAssembly.rotation.z = alphaRad;
      I.leanGroup.quaternion.setFromAxisAngle(I.LEAN_AXIS, leanRad);
      I.pitchGroup.quaternion.setFromAxisAngle(I.PITCH_AXIS, pitchAngle);

      // ── P marker ──
      I.pMarker.position.set(S.px, S.py, 0.005);
      I.pLbl.position.set(S.px, S.py - 0.04, 0.02);

      // ── G marker ──
      I.gMarker.position.set(S.gx, S.gy, S.gz);
      I.gLbl.position.set(S.gx, S.gy + 0.03, S.gz + 0.03);

      // ── L line ──
      setLine(I.lLine, S.px, S.py, 0, S.gx, S.gy, S.gz);

      // ── Gravity arrow ──
      const fgLen = Math.min(Fg / 2000, 0.4);
      setLine(I.fgLine, S.gx, S.gy, S.gz, S.gx, S.gy, S.gz - fgLen);
      I.fgHead.position.set(S.gx, S.gy, S.gz - fgLen);
      I.fgHead.rotation.set(Math.PI, 0, 0);
      I.fgLbl.position.set(S.gx + 0.04, S.gy, S.gz - fgLen * 0.5);

      // ── Velocity arrow ──
      const vx = vel[0];
      const vy = vel[1];
      const vz = vel[2] || 0;
      const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
      I.velLine.line.visible = vMag > 0.1;
      I.velHead.visible = vMag > 0.1;
      I.velLbl.visible = vMag > 0.1;
      if (vMag > 0.1) {
        const vLen = Math.min(speed * 0.05, 0.5);
        const vnx = (vx / vMag) * vLen;
        const vny = (vy / vMag) * vLen;
        const vnz = (vz / vMag) * vLen;
        setLine(I.velLine, S.gx, S.gy, S.gz, S.gx + vnx, S.gy + vny, S.gz + vnz);
        I.velHead.position.set(S.gx + vnx, S.gy + vny, S.gz + vnz);
        const vdir = Math.atan2(vny, vnx);
        I.velHead.rotation.set(Math.PI / 2, 0, -vdir + Math.PI / 2);
        I.velLbl.position.set(S.gx + vnx + 0.03, S.gy + vny, S.gz + vnz + 0.02);
      }

      // ── F_body arrow ──
      const fbLen = Math.min(Fg / 2000, 0.4);
      const gp = new THREE.Vector3(S.px - S.gx, S.py - S.gy, -S.gz).normalize();
      setLine(I.fbLine, S.gx, S.gy, S.gz, S.gx + gp.x * fbLen, S.gy + gp.y * fbLen, S.gz + gp.z * fbLen);
      I.fbLine.line.computeLineDistances();
      I.fbLbl.position.set(S.gx + gp.x * fbLen * 0.5 + 0.04, S.gy + gp.y * fbLen * 0.5, S.gz + gp.z * fbLen * 0.5);

      // ── Resistance along ──
      const raScale = 0.0005;
      const raLen = Math.min(f_along * raScale, 0.15);
      I.raLine.line.visible = raLen > 0.002;
      I.raLbl.visible = raLen > 0.002;
      if (raLen > 0.002) {
        const dir = v_along > 0 ? -1 : 1;
        setLine(I.raLine, S.px, S.py, 0.005, S.px + bladeDir[0] * dir * raLen, S.py + bladeDir[1] * dir * raLen, 0.005);
        I.raLbl.position.set(I.raLine.buf[3] + 0.02, I.raLine.buf[4] + 0.02, 0.02);
      }

      // ── Resistance across ──
      const rcLen = Math.min(f_lateral * raScale, 0.3);
      I.rcLine.line.visible = rcLen > 0.002;
      I.rcLbl.visible = rcLen > 0.002;
      if (rcLen > 0.002) {
        const dir = v_across > 0 ? -1 : 1;
        const perpDir = [-bladeDir[1], bladeDir[0]];
        setLine(I.rcLine, S.px, S.py, 0.005, S.px + perpDir[0] * dir * rcLen, S.py + perpDir[1] * dir * rcLen, 0.005);
        I.rcLbl.position.set(I.rcLine.buf[3] + 0.02, I.rcLine.buf[4] - 0.02, 0.02);
      }

      // ── Theta arc ──
      const arcR = Math.min(L * 0.25, 0.15);
      if (thetaDeg > 1 && dz > 0.01) {
        const dirH = Math.atan2(dy, dx);
        for (let i = 0; i < 33; i++) {
          const t = i / 32;
          const a = t * theta;
          I.thArc.buf[i * 3] = S.px + arcR * Math.sin(a) * Math.cos(dirH);
          I.thArc.buf[i * 3 + 1] = S.py + arcR * Math.sin(a) * Math.sin(dirH);
          I.thArc.buf[i * 3 + 2] = arcR * Math.cos(a);
        }
        I.thArc.geo.attributes.position.needsUpdate = true;
        I.thArc.geo.setDrawRange(0, 33);
        I.thArc.line.visible = true;
        const mid = theta * 0.5;
        I.thLbl.position.set(
          S.px + (arcR + 0.03) * Math.sin(mid) * Math.cos(Math.atan2(dy, dx)),
          S.py + (arcR + 0.03) * Math.sin(mid) * Math.sin(Math.atan2(dy, dx)),
          (arcR + 0.03) * Math.cos(mid)
        );
        I.thLbl.visible = true;
      } else {
        I.thArc.line.visible = false;
        I.thLbl.visible = false;
      }

      // ── Alpha arc ──
      if (S.alpha > 2) {
        const ar = 0.1;
        for (let i = 0; i < 33; i++) {
          const t = i / 32;
          const a = t * alphaRad;
          I.alArc.buf[i * 3] = S.px + ar * Math.cos(a);
          I.alArc.buf[i * 3 + 1] = S.py + ar * Math.sin(a);
          I.alArc.buf[i * 3 + 2] = 0.003;
        }
        I.alArc.geo.attributes.position.needsUpdate = true;
        I.alArc.geo.setDrawRange(0, 33);
        I.alArc.line.visible = true;
        const ma = alphaRad * 0.5;
        I.alLbl.position.set(S.px + (ar + 0.03) * Math.cos(ma), S.py + (ar + 0.03) * Math.sin(ma), 0.01);
        I.alLbl.visible = true;
      } else {
        I.alArc.line.visible = false;
        I.alLbl.visible = false;
      }

      // ── Travel direction ──
      setLine(I.tvLine, S.px, S.py, 0.003, S.px + 0.2, S.py, 0.003);

      // ── Arc path ──
      const nPts = 201;
      if (S.lean > 0.5 && speed > 0.1 && isFinite(R_turn) && R_turn < 100) {
        const R = R_turn;
        const arcLen = Math.min(speed * 3, 2 * Math.PI * R);
        const totalA = arcLen / R;
        for (let i = 0; i < nPts; i++) {
          const t = i / (nPts - 1);
          const a = t * totalA;
          I.arcPath.buf[i * 3] =
            S.px + R * Math.sin(a) * Math.cos(alphaRad) - R * (1 - Math.cos(a)) * Math.sin(alphaRad);
          I.arcPath.buf[i * 3 + 1] =
            S.py + R * Math.sin(a) * Math.sin(alphaRad) + R * (1 - Math.cos(a)) * Math.cos(alphaRad);
          I.arcPath.buf[i * 3 + 2] = 0.002;
        }
        I.arcPath.geo.attributes.position.needsUpdate = true;
        I.arcPath.geo.setDrawRange(0, nPts);
        I.arcPath.line.computeLineDistances();
        I.arcPath.line.visible = true;

        setLine(I.rLine, S.px, S.py, 0.002, S.px - R * Math.sin(alphaRad), S.py + R * Math.cos(alphaRad), 0.002);
        I.rLine.line.computeLineDistances();
        I.rLine.line.visible = true;
        I.rLbl.position.set(
          (I.rLine.buf[0] + I.rLine.buf[3]) / 2 + 0.03,
          (I.rLine.buf[1] + I.rLine.buf[4]) / 2,
          0.02
        );
        I.rLbl.visible = true;
      } else if (speed > 0.1) {
        for (let i = 0; i < nPts; i++) {
          const t = i / (nPts - 1);
          I.arcPath.buf[i * 3] = S.px + t * 2 * Math.cos(alphaRad);
          I.arcPath.buf[i * 3 + 1] = S.py + t * 2 * Math.sin(alphaRad);
          I.arcPath.buf[i * 3 + 2] = 0.002;
        }
        I.arcPath.geo.attributes.position.needsUpdate = true;
        I.arcPath.geo.setDrawRange(0, nPts);
        I.arcPath.line.computeLineDistances();
        I.arcPath.line.visible = true;
        I.rLine.line.visible = false;
        I.rLbl.visible = false;
      } else {
        I.arcPath.line.visible = false;
        I.rLine.line.visible = false;
        I.rLbl.visible = false;
      }
    },

    /** Apply a preset camera position. */
    setCamera(mode, params) {
      const I = internalsRef.current;
      if (!I) return;
      const S = params;
      const cx = (S.gx + S.px) / 2;
      const cy = (S.gy + S.py) / 2;
      const cz = S.gz / 2;
      switch (mode) {
        case 'persp':
          I.camera.position.set(cx + 0.8, cy - 1.2, cz + 0.5);
          I.controls.target.set(cx, cy, cz);
          break;
        case 'side':
          I.camera.position.set(cx, cy - 1.5, cz);
          I.controls.target.set(cx, cy, cz);
          break;
        case 'front':
          I.camera.position.set(cx + 1.5, cy, cz);
          I.controls.target.set(cx, cy, cz);
          break;
        case 'top':
          I.camera.position.set(cx, cy, 2.5);
          I.controls.target.set(cx, cy, 0);
          break;
        case 'blade':
          I.camera.position.set(S.px + 0.06, S.py - 0.12, 0.0);
          I.camera.fov = 30;
          I.camera.near = 0.0001;
          I.camera.updateProjectionMatrix();
          I.controls.target.set(S.px, S.py, 0.012);
          break;
      }
      I.controls.update();
    },
  }));

  return <div ref={mountRef} style={{ marginLeft: PANEL_WIDTH }} />;
});

// ── Utility ──
function setLine(lineObj, x0, y0, z0, x1, y1, z1) {
  lineObj.buf[0] = x0;
  lineObj.buf[1] = y0;
  lineObj.buf[2] = z0;
  lineObj.buf[3] = x1;
  lineObj.buf[4] = y1;
  lineObj.buf[5] = z1;
  lineObj.geo.attributes.position.needsUpdate = true;
}

export default ThreeScene;

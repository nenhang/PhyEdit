/**
 * Match the backend view-to-camera conversion: P_cam = (x, -y_vis, -z_vis),
 * then project with K and use camera-space Z for the depth buffer.
 */
export function projectPointsToImageData(
  positions,
  colors,
  _highlightFlags,
  K9,
  outW,
  outH
) {
  const depthBuf = new Float32Array(outW * outH);
  depthBuf.fill(Infinity);
  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let p = 0; p < out.length; p += 4) {
    out[p] = 20;
    out[p + 1] = 20;
    out[p + 2] = 22;
    out[p + 3] = 255;
  }

  const k = K9;
  const n = positions.length / 3;

  for (let i = 0; i < n; i++) {
    let x = positions[i * 3];
    let y = positions[i * 3 + 1];
    let z = positions[i * 3 + 2];
    const X = x;
    const Y = -y;
    const Z = -z;
    if (Z <= 1e-7) continue;

    const sx = k[0] * X + k[1] * Y + k[2] * Z;
    const sy = k[3] * X + k[4] * Y + k[5] * Z;
    const sz = k[6] * X + k[7] * Y + k[8] * Z;
    if (sz <= 1e-7) continue;

    let u = sx / sz;
    let v = sy / sz;
    const ui = Math.round(u);
    const vi = Math.round(v);
    if (ui < 0 || ui >= outW || vi < 0 || vi >= outH) continue;

    const di = vi * outW + ui;
    if (Z >= depthBuf[di]) continue;
    depthBuf[di] = Z;

    const o = di * 4;
    out[o] = Math.min(255, Math.max(0, colors[i * 3] * 255));
    out[o + 1] = Math.min(255, Math.max(0, colors[i * 3 + 1] * 255));
    out[o + 2] = Math.min(255, Math.max(0, colors[i * 3 + 2] * 255));
    out[o + 3] = 255;
  }
  return out;
}

/** Scale full-resolution intrinsics to the output canvas. */
export function scaleIntrinsicsForSize(K9, fullW, fullH, outW, outH) {
  const sx = outW / fullW;
  const sy = outH / fullH;
  return [
    K9[0] * sx,
    K9[1] * sx,
    K9[2] * sx,
    K9[3] * sy,
    K9[4] * sy,
    K9[5] * sy,
    K9[6],
    K9[7],
    K9[8],
  ];
}

const PC_STRIDE = 8;

function getPointCloudSamplePlan(raw, numPoints, options) {
  const maxPoints = Number(options?.maxPoints || 0);
  if (!maxPoints || numPoints <= maxPoints) {
    return { count: numPoints, step: 1, extraStep: 1, linear: false };
  }

  const imageWidth = Number(options?.imageWidth || 0);
  const imageHeight = Number(options?.imageHeight || 0);
  const maxImageSide = Number(options?.maxImageSide || 0);
  const sideStep =
    imageWidth > 0 && imageHeight > 0 && maxImageSide > 0
      ? Math.max(1, Math.ceil(Math.max(imageWidth, imageHeight) / maxImageSide))
      : 1;
  const countStep = Math.max(1, Math.ceil(Math.sqrt(numPoints / maxPoints)));
  const step = Math.max(sideStep, countStep);

  let count = 0;
  if (imageWidth > 0 && imageHeight > 0) {
    for (let i = 0; i < numPoints; i++) {
      const o = i * PC_STRIDE;
      const u = Math.round(raw[o + 6]);
      const v = Math.round(raw[o + 7]);
      if (u % step === 0 && v % step === 0) count += 1;
    }
  } else {
    count = Math.ceil(numPoints / step);
  }

  if (count < 1) {
    return {
      count: Math.min(numPoints, maxPoints),
      step: Math.max(1, Math.ceil(numPoints / maxPoints)),
      extraStep: 1,
      linear: true,
    };
  }
  return { count, step, extraStep: count > maxPoints ? Math.ceil(count / maxPoints) : 1, linear: false };
}

export function parsePointCloudBin(buffer, options = {}) {
  const raw = new Float32Array(buffer);
  const totalPoints = Math.floor(raw.length / PC_STRIDE);
  const plan = getPointCloudSamplePlan(raw, totalPoints, options);
  const sampledCount = Math.ceil(plan.count / plan.extraStep);
  const positions = new Float32Array(sampledCount * 3);
  const colors = new Float32Array(sampledCount * 3);
  const uv = new Float32Array(sampledCount * 2);

  let kept = 0;
  let seen = 0;
  for (let i = 0; i < totalPoints && kept < sampledCount; i++) {
    const o = i * PC_STRIDE;
    const useGrid =
      !plan.linear && Number(options?.imageWidth || 0) > 0 && Number(options?.imageHeight || 0) > 0;
    let keep = true;
    if (plan.step > 1) {
      if (useGrid) {
        const u = Math.round(raw[o + 6]);
        const v = Math.round(raw[o + 7]);
        keep = u % plan.step === 0 && v % plan.step === 0;
      } else {
        keep = i % plan.step === 0;
      }
    }
    if (!keep) continue;
    if (seen % plan.extraStep !== 0) {
      seen += 1;
      continue;
    }
    seen += 1;

    positions[kept * 3] = raw[o];
    positions[kept * 3 + 1] = raw[o + 1];
    positions[kept * 3 + 2] = raw[o + 2];
    colors[kept * 3] = raw[o + 3];
    colors[kept * 3 + 1] = raw[o + 4];
    colors[kept * 3 + 2] = raw[o + 5];
    uv[kept * 2] = raw[o + 6];
    uv[kept * 2 + 1] = raw[o + 7];
    kept += 1;
  }
  return {
    positions: kept === sampledCount ? positions : positions.slice(0, kept * 3),
    colors: kept === sampledCount ? colors : colors.slice(0, kept * 3),
    uv: kept === sampledCount ? uv : uv.slice(0, kept * 2),
    numPoints: kept,
    totalPoints,
    sampled: kept < totalPoints,
    sampleStep: plan.step,
  };
}

export function maskFlagsFromUv(uv, numPoints, maskW, maskH, lum) {
  const flags = new Uint8Array(numPoints);
  for (let i = 0; i < numPoints; i++) {
    const u = Math.round(uv[i * 2]);
    const v = Math.round(uv[i * 2 + 1]);
    if (u >= 0 && u < maskW && v >= 0 && v < maskH && lum[v * maskW + u]) {
      flags[i] = 1;
    }
  }
  return flags;
}

export function maskCenterFromFlagsUv(uv, flags) {
  if (!uv || !flags) return null;
  let sx = 0;
  let sy = 0;
  let cnt = 0;
  const n = Math.min(flags.length, Math.floor(uv.length / 2));
  for (let i = 0; i < n; i++) {
    if (!flags[i]) continue;
    sx += uv[i * 2];
    sy += uv[i * 2 + 1];
    cnt += 1;
  }
  if (cnt < 1) return null;
  return { cx: sx / cnt, cy: sy / cnt };
}

function invert3x3(m) {
  const a = m[0], b = m[1], c = m[2];
  const d = m[3], e = m[4], f = m[5];
  const g = m[6], h = m[7], i = m[8];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1.0 / det;
  return [A * invDet, D * invDet, G * invDet, B * invDet, E * invDet, H * invDet, C * invDet, F * invDet, I * invDet];
}

function rotateUv(u, v, cx, cy, angleDeg) {
  if (!Number.isFinite(angleDeg) || Math.abs(angleDeg) < 1e-8) return [u, v];
  const t = (angleDeg * Math.PI) / 180.0;
  const ct = Math.cos(t);
  const st = Math.sin(t);
  const du = u - cx;
  const dv = v - cy;
  return [cx + du * ct - dv * st, cy + du * st + dv * ct];
}

function rotateAroundImageAxesCamera(X, Y, Z, center, K9, rotVerticalDeg, rotHorizontalDeg, centerDepthZ) {
  if (!center || !K9) return [X, Y, Z];
  const rv = Number(rotVerticalDeg || 0);
  const rh = Number(rotHorizontalDeg || 0);
  if (Math.abs(rv) < 1e-8 && Math.abs(rh) < 1e-8) return [X, Y, Z];
  const fx = K9[0];
  const fy = K9[4];
  const cx = K9[2];
  const cy = K9[5];
  const zc = Math.max(centerDepthZ || 1, 1e-6);
  const Cx = ((center.cx - cx) * zc) / fx;
  const Cy = ((center.cy - cy) * zc) / fy;
  const Cz = zc;
  let x = X - Cx;
  let y = Y - Cy;
  let z = Z - Cz;
  if (Math.abs(rv) > 1e-8) {
    const t = (rv * Math.PI) / 180.0;
    const ct = Math.cos(t);
    const st = Math.sin(t);
    const xn = ct * x + st * z;
    const zn = -st * x + ct * z;
    x = xn;
    z = zn;
  }
  if (Math.abs(rh) > 1e-8) {
    const t = (rh * Math.PI) / 180.0;
    const ct = Math.cos(t);
    const st = Math.sin(t);
    const yn = ct * y - st * z;
    const zn = st * y + ct * z;
    y = yn;
    z = zn;
  }
  return [x + Cx, y + Cy, z + Cz];
}

function transformSelectedPointBy2DRotationVis(
  x,
  y,
  z,
  selected,
  translation,
  rotationDeg,
  rotationVerticalDeg,
  rotationHorizontalDeg,
  center,
  K9,
  invK,
  imageWidth,
  imageHeight,
  zLock
) {
  if (!selected) return [x, y, z];
  const duPx = (translation?.x ?? 0) * (imageWidth - 1);
  const dvPx = (translation?.y ?? 0) * (imageHeight - 1);
  const tz = translation?.z ?? 0;
  const xv = x;
  const yv = y;
  const zv = z - tz;
  if (!K9 || !invK || !center) return [xv, yv, zv];

  const X = xv;
  const Y = -yv;
  const Z = -zv;
  if (Z <= 1e-8) return [xv, yv, zv];
  const [Xr3, Yr3, Zr3] = rotateAroundImageAxesCamera(
    X,
    Y,
    Z,
    center,
    K9,
    rotationVerticalDeg,
    rotationHorizontalDeg,
    zLock?.centerDepthZ
  );
  const sx = K9[0] * Xr3 + K9[1] * Yr3 + K9[2] * Zr3;
  const sy = K9[3] * Xr3 + K9[4] * Yr3 + K9[5] * Zr3;
  const sz = K9[6] * Xr3 + K9[7] * Yr3 + K9[8] * Zr3;
  if (Math.abs(sz) <= 1e-8) return [xv, yv, zv];
  let u = sx / sz;
  let v = sy / sz;
  if (zLock && center && K9 && Math.abs(tz) > 1e-8) {
    const cx0 = K9[2];
    const cy0 = K9[5];
    const zBefore = Math.max(zLock.centerDepthZ, 1e-6);
    const zAfter = Math.max(zBefore + tz, 1e-6);
    const uAfterCenter = ((center.cx - cx0) * zBefore) / zAfter + cx0;
    const vAfterCenter = ((center.cy - cy0) * zBefore) / zAfter + cy0;
    u += center.cx - uAfterCenter;
    v += center.cy - vAfterCenter;
  }
  u += duPx;
  v += dvPx;
  [u, v] = rotateUv(u, v, center.cx + duPx, center.cy + dvPx, rotationDeg);

  const nx = invK[0] * u + invK[1] * v + invK[2];
  const ny = invK[3] * u + invK[4] * v + invK[5];
  const nz = invK[6] * u + invK[7] * v + invK[8];
  if (Math.abs(nz) <= 1e-8) return [xv, yv, zv];
  const Xr = (nx / nz) * Zr3;
  const Yr = (ny / nz) * Zr3;
  const Zr = Zr3;
  return [Xr, -Yr, -Zr];
}

export function applySelectionTransformToPositions(
  positions,
  flags,
  translation,
  rotationDeg,
  rotationVerticalDeg,
  rotationHorizontalDeg,
  center,
  K9,
  imageWidth,
  imageHeight
) {
  const out = new Float32Array(positions.length);
  const invK = K9 ? invert3x3(K9) : null;
  const n = Math.floor(positions.length / 3);
  const hasFlags = flags && flags.length === n;
  let centerDepthZ = 1.0;
  if (center && hasFlags) {
    let s = 0;
    let c = 0;
    for (let i = 0; i < n; i++) {
      if (!flags[i]) continue;
      const zVis = positions[i * 3 + 2];
      const zCam = -zVis;
      if (zCam > 1e-6) {
        s += zCam;
        c += 1;
      }
    }
    if (c > 0) centerDepthZ = s / c;
  }
  const zLock = { centerDepthZ };
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const sel = hasFlags && !!flags[i];
    const [xo, yo, zo] = transformSelectedPointBy2DRotationVis(
      x,
      y,
      z,
      sel,
      translation,
      rotationDeg,
      rotationVerticalDeg,
      rotationHorizontalDeg,
      center,
      K9,
      invK,
      imageWidth,
      imageHeight,
      zLock
    );
    out[i * 3] = xo;
    out[i * 3 + 1] = yo;
    out[i * 3 + 2] = zo;
  }
  return out;
}

export function applyObjectTransformsToPositions(positions, objectTransforms, K9, imageWidth, imageHeight) {
  let out = new Float32Array(positions);
  if (!objectTransforms || objectTransforms.length === 0) return out;
  for (const obj of objectTransforms) {
    out = applySelectionTransformToPositions(
      out,
      obj.flags,
      obj.translation ?? { x: 0, y: 0, z: 0 },
      obj.rotationDeg ?? 0,
      obj.rotationVerticalDeg ?? 0,
      obj.rotationHorizontalDeg ?? 0,
      obj.center ?? null,
      K9,
      imageWidth,
      imageHeight
    );
  }
  return out;
}

export function mergeFlags(flagsList, n) {
  const out = new Uint8Array(n);
  for (const f of flagsList || []) {
    if (!f) continue;
    const m = Math.min(n, f.length);
    for (let i = 0; i < m; i++) {
      if (f[i]) out[i] = 1;
    }
  }
  return out;
}

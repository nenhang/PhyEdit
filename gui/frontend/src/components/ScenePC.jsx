import React, { useCallback, useEffect, useRef } from "react";
import { OrbitControls, TransformControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import {
  applyObjectTransformsToPositions,
  mergeFlags,
  maskCenterFromFlagsUv,
  maskFlagsFromUv,
  parsePointCloudBin,
} from "../utils/pointCloudProject";

const VIEW_PC_MAX_POINTS = 90000;
const VIEW_PC_MAX_SIDE = 480;

function boundsFromPositions(positions, flags = null) {
  if (!positions || positions.length < 3) return null;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  const n = Math.floor(positions.length / 3);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (flags && !flags[i]) continue;
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
    count += 1;
  }
  if (count === 0) return null;
  const center = new THREE.Vector3(
    (minX + maxX) * 0.5,
    (minY + maxY) * 0.5,
    (minZ + maxZ) * 0.5
  );
  const size = new THREE.Vector3(maxX - minX, maxY - minY, maxZ - minZ);
  const radius = Math.max(size.length() * 0.5, 0.05);
  return { center, radius, count };
}

/** Point-cloud binary rows: x, y, z, r, g, b, u, v. */
export default function ScenePC({
  pcUrl,
  objects,
  selectedObjectId,
  intrinsics,
  imageWidth,
  imageHeight,
  apiBase,
  viewCommand,
  onSelectedTranslationChange,
}) {
  const { camera } = useThree();
  const cameraRef = useRef(camera);
  const controlsRef = useRef(null);
  const gizmoRef = useRef(null);
  const gizmoDragRef = useRef(null);
  const geometryRef = useRef(null);
  const basePositionsRef = useRef(null);
  const baseColorsRef = useRef(null);
  const uvRef = useRef(null);
  const maskTransformCacheRef = useRef(new Map());
  const objectTransformsRef = useRef([]);
  const sceneBoundsRef = useRef(null);
  const selectedBoundsRef = useRef(null);
  const [pointSize, setPointSize] = React.useState(1.8);
  const [pcTick, setPcTick] = React.useState(0);
  const selectedObject = objects?.find((obj) => obj.object_id === selectedObjectId) || null;

  useEffect(() => {
    cameraRef.current = camera;
  }, [camera]);

  const focusBounds = useCallback(
    (bounds, padding = 1.25) => {
      if (!bounds) return;
      const activeCamera = cameraRef.current;
      if (!activeCamera) return;
      const controls = controlsRef.current;
      const target = bounds.center.clone();
      const oldTarget = controls?.target?.clone?.() || new THREE.Vector3(0, 0, 0);
      const direction = new THREE.Vector3().subVectors(activeCamera.position, oldTarget);
      if (!Number.isFinite(direction.lengthSq()) || direction.lengthSq() < 1e-8) {
        direction.set(0, 0, 1);
      }
      direction.normalize();

      const fov = THREE.MathUtils.degToRad(activeCamera.fov || 50);
      const distance = Math.max(bounds.radius / Math.sin(fov * 0.5), bounds.radius * 2.2, 0.2) * padding;
      activeCamera.position.copy(target).addScaledVector(direction, distance);
      activeCamera.near = Math.max(distance / 10000, 0.001);
      activeCamera.far = Math.max(distance + bounds.radius * 12, distance * 12, 1000);
      activeCamera.updateProjectionMatrix();

      if (controls) {
        controls.target.copy(target);
        controls.minDistance = Math.max(bounds.radius * 0.02, 0.001);
        controls.maxDistance = Math.max(bounds.radius * 80, distance * 12, 10);
        controls.update();
      } else {
        activeCamera.lookAt(target);
      }
    },
    []
  );

  const projectViewPoint = useCallback(
    (point) => {
      if (!point || !intrinsics || !imageWidth || !imageHeight) return null;
      const x = point.x;
      const y = -point.y;
      const z = -point.z;
      if (!Number.isFinite(z) || z <= 1e-8) return null;
      const sx = intrinsics[0] * x + intrinsics[1] * y + intrinsics[2] * z;
      const sy = intrinsics[3] * x + intrinsics[4] * y + intrinsics[5] * z;
      const sz = intrinsics[6] * x + intrinsics[7] * y + intrinsics[8] * z;
      if (!Number.isFinite(sz) || Math.abs(sz) <= 1e-8) return null;
      return { u: sx / sz, v: sy / sz, z };
    },
    [intrinsics, imageWidth, imageHeight]
  );

  const syncGizmoToSelection = useCallback(() => {
    if (!gizmoRef.current || gizmoDragRef.current?.active) return;
    const bounds = selectedBoundsRef.current;
    if (!bounds) return;
    gizmoRef.current.position.copy(bounds.center);
  }, []);

  const handleGizmoMouseDown = useCallback(() => {
    if (!gizmoRef.current || !selectedObject) return;
    const center = gizmoRef.current.position.clone();
    gizmoDragRef.current = {
      active: true,
      startPosition: center,
      startPixel: projectViewPoint(center),
      startTranslation: {
        x: Number(selectedObject.translation?.x || 0),
        y: Number(selectedObject.translation?.y || 0),
        z: Number(selectedObject.translation?.z || 0),
      },
    };
  }, [projectViewPoint, selectedObject]);

  const handleGizmoObjectChange = useCallback(() => {
    const drag = gizmoDragRef.current;
    const gizmo = gizmoRef.current;
    if (!drag?.active || !gizmo || typeof onSelectedTranslationChange !== "function") return;

    const current = gizmo.position.clone();
    const currentPixel = projectViewPoint(current);
    const deltaVis = new THREE.Vector3().subVectors(current, drag.startPosition);
    const next = {
      ...drag.startTranslation,
      z: drag.startTranslation.z - deltaVis.z,
    };

    if (drag.startPixel && currentPixel) {
      next.x = drag.startTranslation.x + (currentPixel.u - drag.startPixel.u) / Math.max(imageWidth - 1, 1);
      next.y = drag.startTranslation.y + (currentPixel.v - drag.startPixel.v) / Math.max(imageHeight - 1, 1);
    }
    onSelectedTranslationChange(next);
  }, [imageHeight, imageWidth, onSelectedTranslationChange, projectViewPoint]);

  const handleGizmoMouseUp = useCallback(() => {
    gizmoDragRef.current = null;
    syncGizmoToSelection();
  }, [syncGizmoToSelection]);

  const applyVisuals = useCallback(() => {
    const geo = geometryRef.current;
    const baseP = basePositionsRef.current;
    const baseC = baseColorsRef.current;
    const objectTransforms = objectTransformsRef.current || [];
    if (!geo || !baseP || !baseC) return;

    const pos = geo.attributes.position.array;
    const col = geo.attributes.color.array;
    const n = baseP.length / 3;
    const transformed = applyObjectTransformsToPositions(
      baseP,
      objectTransforms,
      intrinsics,
      imageWidth,
      imageHeight
    );
    const selectedFlags =
      objectTransforms.find((x) => x.object_id === selectedObjectId)?.flags || null;
    const unionFlags = mergeFlags(
      objectTransforms.map((x) => x.flags),
      n
    );

    const HIL_SELECTED = [1, 0.82, 0.15];
    const HIL_OTHER = [0.4, 0.78, 1.0];
    const mixSel = 0.45;
    const mixOther = 0.28;
    const selectedPositions = selectedFlags ? transformed : null;

    for (let i = 0; i < n; i++) {
      const isSel = !!(selectedFlags && selectedFlags[i]);
      const isObj = !!unionFlags[i];
      pos[i * 3] = transformed[i * 3];
      pos[i * 3 + 1] = transformed[i * 3 + 1];
      pos[i * 3 + 2] = transformed[i * 3 + 2];

      if (isSel) {
        col[i * 3] = baseC[i * 3] * (1 - mixSel) + HIL_SELECTED[0] * mixSel;
        col[i * 3 + 1] = baseC[i * 3 + 1] * (1 - mixSel) + HIL_SELECTED[1] * mixSel;
        col[i * 3 + 2] = baseC[i * 3 + 2] * (1 - mixSel) + HIL_SELECTED[2] * mixSel;
      } else if (isObj) {
        col[i * 3] = baseC[i * 3] * (1 - mixOther) + HIL_OTHER[0] * mixOther;
        col[i * 3 + 1] = baseC[i * 3 + 1] * (1 - mixOther) + HIL_OTHER[1] * mixOther;
        col[i * 3 + 2] = baseC[i * 3 + 2] * (1 - mixOther) + HIL_OTHER[2] * mixOther;
      } else {
        col[i * 3] = baseC[i * 3];
        col[i * 3 + 1] = baseC[i * 3 + 1];
        col[i * 3 + 2] = baseC[i * 3 + 2];
      }
    }
    selectedBoundsRef.current = selectedPositions ? boundsFromPositions(selectedPositions, selectedFlags) : null;
    syncGizmoToSelection();
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  }, [selectedObjectId, intrinsics, imageWidth, imageHeight, syncGizmoToSelection]);

  useEffect(() => {
    if (!pcUrl) return;
    basePositionsRef.current = null;
    baseColorsRef.current = null;
    uvRef.current = null;
    maskTransformCacheRef.current.clear();
    objectTransformsRef.current = [];

    fetch(pcUrl)
      .then((res) => res.arrayBuffer())
      .then((buffer) => {
        const { positions, colors, uv } = parsePointCloudBin(buffer, {
          maxPoints: VIEW_PC_MAX_POINTS,
          maxImageSide: VIEW_PC_MAX_SIDE,
          imageWidth,
          imageHeight,
        });
        basePositionsRef.current = new Float32Array(positions);
        baseColorsRef.current = new Float32Array(colors);
        uvRef.current = uv;
        const sceneBounds = boundsFromPositions(positions);
        sceneBoundsRef.current = sceneBounds;
        selectedBoundsRef.current = null;
        if (sceneBounds) {
          const adaptiveSize = THREE.MathUtils.clamp(sceneBounds.radius * 0.004, 1.4, 2.6);
          setPointSize(adaptiveSize);
        }

        const geo = geometryRef.current;
        if (!geo) return;
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geo.attributes.position.needsUpdate = true;
        geo.attributes.color.needsUpdate = true;
        geo.computeBoundingBox();
        geo.computeBoundingSphere();
        setPcTick((t) => t + 1);
        requestAnimationFrame(() => focusBounds(sceneBoundsRef.current, 1.35));
      });
  }, [pcUrl, imageWidth, imageHeight, focusBounds]);

  useEffect(() => {
    if (!objects || objects.length === 0 || !uvRef.current || !basePositionsRef.current) {
      objectTransformsRef.current = [];
      applyVisuals();
      return;
    }
    const uv = uvRef.current;
    const n = uv.length / 2;
    let cancelled = false;

    const loadMaskTransform = (obj) => {
      const fullUrl = obj.mask_url.startsWith("http")
        ? obj.mask_url
        : `${apiBase || ""}${obj.mask_url}`;
      let maskPromise = maskTransformCacheRef.current.get(fullUrl);
      if (!maskPromise) {
        maskPromise = new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
              const mw = img.naturalWidth;
              const mh = img.naturalHeight;
              const canvas = document.createElement("canvas");
              canvas.width = mw;
              canvas.height = mh;
              const ctx = canvas.getContext("2d");
              if (!ctx) return resolve(null);
              ctx.drawImage(img, 0, 0);
              const im = ctx.getImageData(0, 0, mw, mh);
              const lum = new Uint8Array(mw * mh);
              for (let i = 0; i < mw * mh; i++) lum[i] = im.data[i * 4] > 127 ? 1 : 0;
              const flags = maskFlagsFromUv(uv, n, mw, mh, lum);
              resolve({ flags, center: maskCenterFromFlagsUv(uv, flags) });
            };
            img.onerror = () => resolve(null);
            img.src = fullUrl;
          });
        maskTransformCacheRef.current.set(fullUrl, maskPromise);
      }
      return maskPromise.then((maskData) =>
        maskData
          ? {
              object_id: obj.object_id,
              ...maskData,
              translation: obj.translation,
              rotationDeg: obj.rotation_deg,
              rotationVerticalDeg: obj.rotation_v_deg,
              rotationHorizontalDeg: obj.rotation_h_deg,
            }
          : null
      );
    };

    Promise.all(objects.map(loadMaskTransform)).then((arr) => {
      if (cancelled) return;
      objectTransformsRef.current = arr.filter(Boolean);
      applyVisuals();
    });
    return () => {
      cancelled = true;
    };
  }, [objects, pcUrl, apiBase, pcTick, applyVisuals]);

  useEffect(() => {
    if (!viewCommand?.token) return;
    if (viewCommand.mode === "selected" && selectedBoundsRef.current) {
      focusBounds(selectedBoundsRef.current, 1.55);
      return;
    }
    focusBounds(sceneBoundsRef.current, 1.35);
  }, [viewCommand, focusBounds]);

  return (
    <>
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        enablePan
        screenSpacePanning
        zoomToCursor
        rotateSpeed={0.75}
        panSpeed={0.9}
        zoomSpeed={0.9}
        mouseButtons={{
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN,
        }}
      />
      <ambientLight intensity={1} />
      <points frustumCulled={false}>
        <bufferGeometry ref={geometryRef} />
        <pointsMaterial size={pointSize} vertexColors sizeAttenuation={false} />
      </points>
      {selectedObject && (
        <>
          <mesh ref={gizmoRef} visible={false} />
          <TransformControls
            object={gizmoRef}
            mode="translate"
            space="world"
            size={0.82}
            enabled={!!selectedObject}
            onMouseDown={handleGizmoMouseDown}
            onObjectChange={handleGizmoObjectChange}
            onMouseUp={handleGizmoMouseUp}
          />
        </>
      )}
    </>
  );
}

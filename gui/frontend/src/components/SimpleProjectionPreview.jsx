import React, { useCallback, useDeferredValue, useEffect, useRef, useState } from "react";
import { Box, Typography } from "@mui/material";
import {
  applyObjectTransformsToPositions,
  mergeFlags,
  maskCenterFromFlagsUv,
  parsePointCloudBin,
  projectPointsToImageData,
  scaleIntrinsicsForSize,
  maskFlagsFromUv,
} from "../utils/pointCloudProject";

const PREVIEW_MAX_SIDE = 384;
const PREVIEW_MAX_POINTS = 120000;

/** Project points with K and a Z-buffer while bounding browser-side work. */
export default function SimpleProjectionPreview({
  pcUrl,
  previewImageUrl,
  objects,
  selectedObjectId,
  intrinsics,
  imageWidth,
  imageHeight,
  apiBase,
}) {
  const deferredObjects = useDeferredValue(objects || []);
  const canvasRef = useRef(null);
  const pcRef = useRef(null);
  const [pcRevision, setPcRevision] = useState(0);
  const [maskCache, setMaskCache] = useState({});
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    pcRef.current = null;
    if (!pcUrl || previewImageUrl) return;

    let cancelled = false;
    fetch(pcUrl)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        if (cancelled) return;
        pcRef.current = parsePointCloudBin(buf, {
          maxPoints: PREVIEW_MAX_POINTS,
          maxImageSide: PREVIEW_MAX_SIDE,
          imageWidth,
          imageHeight,
        });
        setLoadError(null);
        setPcRevision((x) => x + 1);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e.message || e));
      });
    return () => {
      cancelled = true;
    };
  }, [pcUrl, previewImageUrl, imageWidth, imageHeight]);

  useEffect(() => {
    if (previewImageUrl || !deferredObjects || deferredObjects.length === 0) {
      return;
    }
    Promise.all(
      deferredObjects.map(
        (obj) =>
          new Promise((resolve) => {
            const fullUrl = obj.mask_url.startsWith("http") ? obj.mask_url : `${apiBase || ""}${obj.mask_url}`;
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
              const mw = img.naturalWidth;
              const mh = img.naturalHeight;
              const c = document.createElement("canvas");
              c.width = mw;
              c.height = mh;
              const ctx = c.getContext("2d");
              if (!ctx) return resolve([obj.object_id, null]);
              ctx.drawImage(img, 0, 0);
              const im = ctx.getImageData(0, 0, mw, mh);
              const lum = new Uint8Array(mw * mh);
              for (let i = 0; i < mw * mh; i++) lum[i] = im.data[i * 4] > 127 ? 1 : 0;
              resolve([obj.object_id, { mw, mh, lum }]);
            };
            img.onerror = () => resolve([obj.object_id, null]);
            img.src = fullUrl;
          })
      )
    ).then((entries) => {
      const next = {};
      for (const [k, v] of entries) if (v) next[k] = v;
      setMaskCache(next);
    });
  }, [deferredObjects, apiBase, previewImageUrl]);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const pc = pcRef.current;
    if (previewImageUrl || !canvas || !pc || !intrinsics || !imageWidth || !imageHeight) return;

    const maxSide = Math.max(imageWidth, imageHeight);
    const scale = maxSide > PREVIEW_MAX_SIDE ? PREVIEW_MAX_SIDE / maxSide : 1;
    const outW = Math.max(1, Math.round(imageWidth * scale));
    const outH = Math.max(1, Math.round(imageHeight * scale));

    const Ks = scaleIntrinsicsForSize(intrinsics, imageWidth, imageHeight, outW, outH);

    const n = pc.numPoints;
    const transforms = (deferredObjects || [])
      .map((obj) => {
        const mk = maskCache[obj.object_id];
        if (!mk) return null;
        const flags = maskFlagsFromUv(pc.uv, n, mk.mw, mk.mh, mk.lum);
        return {
          object_id: obj.object_id,
          flags,
          center: maskCenterFromFlagsUv(pc.uv, flags),
          translation: obj.translation,
          rotationDeg: obj.rotation_deg,
          rotationVerticalDeg: obj.rotation_v_deg,
          rotationHorizontalDeg: obj.rotation_h_deg,
        };
      })
      .filter(Boolean);
    const transformed = applyObjectTransformsToPositions(
      pc.positions,
      transforms,
      intrinsics,
      imageWidth,
      imageHeight
    );
    const selectedFlags = transforms.find((x) => x.object_id === selectedObjectId)?.flags || null;
    const unionFlags = mergeFlags(transforms.map((x) => x.flags), n);
    const imgData = projectPointsToImageData(
      transformed,
      pc.colors,
      selectedFlags || unionFlags,
      Ks,
      outW,
      outH
    );

    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(imgData, outW, outH), 0, 0);
  }, [
    intrinsics,
    imageWidth,
    imageHeight,
    deferredObjects,
    selectedObjectId,
    maskCache,
    previewImageUrl,
  ]);

  useEffect(() => {
    paint();
  }, [paint, pcRevision]);

  if (previewImageUrl) {
    return (
      <Box sx={{ mt: 0.5 }}>
        <Typography variant="caption" sx={{ color: "#8b98aa", display: "block", mb: 0.5 }}>
          Backend preview rendered from the full point cloud
        </Typography>
        <Box
          component="img"
          src={previewImageUrl}
          alt="point-cloud-preview"
          sx={{
            width: "100%",
            maxWidth: "100%",
            height: "auto",
            display: "block",
            borderRadius: "8px",
            border: "1px solid rgba(148, 163, 184, 0.16)",
            bgcolor: "#05080d",
          }}
        />
      </Box>
    );
  }

  if (!pcUrl || !intrinsics?.length) {
    return null;
  }

  return (
    <Box sx={{ mt: 0.5 }}>
      <Typography variant="caption" sx={{ color: "#8b98aa", display: "block", mb: 0.5 }}>
        Live projection (Z-buffer, {PREVIEW_MAX_SIDE}px maximum side, up to {PREVIEW_MAX_POINTS.toLocaleString()} points)
      </Typography>
      {loadError && (
        <Typography variant="caption" color="error">
          Failed to load the point cloud: {loadError}
        </Typography>
      )}
      <Box
        component="canvas"
        ref={canvasRef}
        sx={{
          width: "100%",
          maxWidth: "100%",
          height: "auto",
          display: "block",
          borderRadius: "8px",
          border: "1px solid rgba(148, 163, 184, 0.16)",
          bgcolor: "#05080d",
          imageRendering: "pixelated",
        }}
      />
    </Box>
  );
}

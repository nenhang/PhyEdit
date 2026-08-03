import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Box,
  Button,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";

/** Overlay masks and positive/negative SAM boxes on the source image. */
export default function ImageSegmentPanel({
  imageSrc,
  disabled,
  onRunSegment,
  boxDrawMode,
  onBoxDrawModeChange,
  showControls = true,
  objects = [],
  selectedObjectId = null,
  apiBase,
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [textPrompt, setTextPrompt] = useState("");
  const [boxes, setBoxes] = useState([]);
  const dragRef = useRef(null);
  const maskOverlayRef = useRef({});
  const maskLoadCacheRef = useRef(new Map());
  const objectOrderRef = useRef([]);
  const objectsRef = useRef(objects);
  const [maskVer, setMaskVer] = useState(0);
  const maskSourcesKey = (objects || [])
    .map((obj) => `${obj.object_id}:${obj.mask_url || ""}`)
    .join("|");
  const fieldSx = {
    "& .MuiInputBase-root": {
      color: "#e8eef7",
      bgcolor: "#0b1118",
      borderRadius: "6px",
    },
    "& .MuiInputBase-input::placeholder": {
      color: "#6f7f93",
      opacity: 1,
    },
    "& .MuiInputLabel-root": { color: "#8492a6" },
    "& .MuiInputLabel-root.Mui-focused": { color: "#8cc8ff" },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(148, 163, 184, 0.22)" },
    "& .MuiInputBase-root:hover .MuiOutlinedInput-notchedOutline": {
      borderColor: "rgba(148, 163, 184, 0.42)",
    },
    "& .Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#4ea8ff" },
  };
  const buttonSx = { borderRadius: "6px", fontWeight: 700 };

  useEffect(() => {
    objectsRef.current = objects;
  }, [objects]);

  useEffect(() => {
    let cancelled = false;
    const maskObjects = (objectsRef.current || []).map((obj) => ({
      object_id: obj.object_id,
      mask_url: obj.mask_url,
    }));
    objectOrderRef.current = maskObjects.map((obj) => obj.object_id);
    if (maskObjects.length === 0) {
      maskOverlayRef.current = {};
      return () => {
        cancelled = true;
      };
    }

    Promise.all(
      maskObjects.map((obj) => {
        const maskUrl = obj?.mask_url;
        if (!maskUrl) return Promise.resolve([obj.object_id, null]);
        const fullUrl = maskUrl.startsWith("http") ? maskUrl : `${apiBase || ""}${maskUrl}`;
        let maskPromise = maskLoadCacheRef.current.get(fullUrl);
        if (!maskPromise) {
          maskPromise = new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
              const mw = img.naturalWidth;
              const mh = img.naturalHeight;
              const c = document.createElement("canvas");
              c.width = mw;
              c.height = mh;
              const ctx = c.getContext("2d");
              if (!ctx) return resolve(null);
              ctx.drawImage(img, 0, 0);
              const im = ctx.getImageData(0, 0, mw, mh);
              const lum = new Uint8Array(mw * mh);
              for (let i = 0; i < mw * mh; i++) lum[i] = im.data[i * 4] > 127 ? 1 : 0;
              resolve({ lum, mw, mh });
            };
            img.onerror = () => resolve(null);
            img.src = fullUrl;
          });
          maskLoadCacheRef.current.set(fullUrl, maskPromise);
        }
        return maskPromise.then((maskData) => [obj.object_id, maskData]);
      })
    ).then((entries) => {
      if (cancelled) return;
      const cache = {};
      for (const [k, v] of entries) if (v) cache[k] = v;
      maskOverlayRef.current = cache;
      setMaskVer((v) => v + 1);
    });
    return () => {
      cancelled = true;
    };
  }, [maskSourcesKey, apiBase]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const img = wrap.querySelector("img");
    if (!img || !img.complete || !img.naturalWidth) return;

    const rect = wrap.getBoundingClientRect();
    const scale = rect.width / img.naturalWidth;
    const dispH = img.naturalHeight * scale;
    canvas.width = rect.width;
    canvas.height = dispH;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${dispH}px`;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const overlays = maskOverlayRef.current || {};
    const palette = [
      [0, 200, 255],
      [20, 230, 120],
      [255, 170, 0],
      [200, 140, 255],
      [255, 100, 140],
    ];
    const ordered = maskSourcesKey
      ? objectOrderRef.current.filter((id) => overlays[id])
      : [];
    for (let k = 0; k < ordered.length; k++) {
      const id = ordered[k];
      const mo = overlays[id];
      const sm = document.createElement("canvas");
      sm.width = mo.mw;
      sm.height = mo.mh;
      const sctx = sm.getContext("2d");
      if (!sctx) continue;
      const sd = sctx.createImageData(mo.mw, mo.mh);
      const rgb = palette[k % palette.length];
      const alpha = id === selectedObjectId ? 135 : 85;
      for (let i = 0; i < mo.mw * mo.mh; i++) {
        if (!mo.lum[i]) continue;
        const o = i * 4;
        sd.data[o] = rgb[0];
        sd.data[o + 1] = rgb[1];
        sd.data[o + 2] = rgb[2];
        sd.data[o + 3] = alpha;
      }
      sctx.putImageData(sd, 0, 0);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(sm, 0, 0, mo.mw, mo.mh, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }

    boxes.forEach((b) => {
      const [x1, y1, x2, y2] = b.xyxy.map((p) => p * scale);
      ctx.strokeStyle = b.label === 1 ? "#00e676" : "#ff5252";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
    const d = dragRef.current;
    if (d) {
      const sx = d.sx * scale;
      const sy = d.sy * scale;
      const ex = d.ex * scale;
      const ey = d.ey * scale;
      ctx.strokeStyle = boxDrawMode === "positive" ? "#69f0ae" : "#ff8a80";
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(sx, sy, ex - sx, ey - sy);
      ctx.setLineDash([]);
    }
  }, [boxes, boxDrawMode, selectedObjectId, maskSourcesKey]);

  React.useEffect(() => {
    redraw();
  }, [redraw, imageSrc, boxes, maskVer]);

  React.useEffect(() => {
    const ro = new ResizeObserver(() => redraw());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [redraw]);

  const toNatural = (clientX, clientY) => {
    const wrap = wrapRef.current;
    const im = wrap?.querySelector("img");
    if (!wrap || !im?.naturalWidth) return null;
    const r = wrap.getBoundingClientRect();
    const sc = im.naturalWidth / r.width;
    const x = (clientX - r.left) * sc;
    const y = (clientY - r.top) * sc;
    return { x, y, nw: im.naturalWidth, nh: im.naturalHeight };
  };

  const onPointerDown = (e) => {
    if (disabled) return;
    if (e.button !== undefined && e.button !== 0) return;
    const p = toNatural(e.clientX, e.clientY);
    if (!p) return;
    e.preventDefault();
    e.stopPropagation();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    dragRef.current = { sx: p.x, sy: p.y, ex: p.x, ey: p.y, nw: p.nw, nh: p.nh, pointerId: e.pointerId };
    redraw();
  };

  const onPointerMove = (e) => {
    if (!dragRef.current) return;
    const p = toNatural(e.clientX, e.clientY);
    if (!p) return;
    dragRef.current.ex = p.x;
    dragRef.current.ey = p.y;
    redraw();
  };

  const finishPointer = (e) => {
    const d = dragRef.current;
    if (d?.pointerId != null && e?.currentTarget) {
      try {
        if (e.currentTarget.hasPointerCapture?.(d.pointerId)) {
          e.currentTarget.releasePointerCapture(d.pointerId);
        }
      } catch {
        /* ignore */
      }
    }
    dragRef.current = null;
    if (!d) return;
    let x1 = Math.min(d.sx, d.ex);
    let y1 = Math.min(d.sy, d.ey);
    let x2 = Math.max(d.sx, d.ex);
    let y2 = Math.max(d.sy, d.ey);
    if (Math.abs(x2 - x1) < 4 || Math.abs(y2 - y1) < 4) {
      redraw();
      return;
    }
    x1 = Math.max(0, Math.min(d.nw - 1, x1));
    x2 = Math.max(0, Math.min(d.nw - 1, x2));
    y1 = Math.max(0, Math.min(d.nh - 1, y1));
    y2 = Math.max(0, Math.min(d.nh - 1, y2));
    const label = boxDrawMode === "positive" ? 1 : 0;
    setBoxes((prev) => [...prev, { xyxy: [x1, y1, x2, y2], label }]);
    redraw();
  };

  const onPointerUp = (e) => finishPointer(e);

  const onPointerCancel = (e) => finishPointer(e);

  const clearBoxes = () => setBoxes([]);

  const handleSegment = () => {
    const payload = { text: textPrompt.trim() || undefined };
    if (boxes.length > 0) {
      const bx = boxes.map((b) => b.xyxy);
      payload.input_boxes = [bx];
      payload.input_boxes_labels = [boxes.map((b) => b.label)];
    }
    onRunSegment(payload);
  };

  return (
    <Stack spacing={1.35} sx={{ width: "100%", height: "100%" }}>
      {showControls && (
        <>
          <Typography variant="subtitle2" sx={{ color: "#c8d3e2", fontWeight: 800 }}>
            SAM prompts
          </Typography>
          <TextField
            size="small"
            label="Text prompt (optional)"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            disabled={disabled}
            fullWidth
            placeholder="e.g. handle"
            sx={fieldSx}
          />
          <Box>
            <Typography variant="caption" sx={{ color: "#8b98aa", display: "block", mb: 0.5 }}>
              Box type
            </Typography>
            <ToggleButtonGroup
              value={boxDrawMode}
              exclusive
              onChange={(_, v) => v != null && onBoxDrawModeChange(v)}
              size="small"
              disabled={disabled}
              sx={{
                flexWrap: "wrap",
                "& .MuiToggleButton-root": {
                  color: "#b8c4d6",
                  borderColor: "rgba(148, 163, 184, 0.24)",
                  px: 1.4,
                  py: 0.55,
                },
                "& .MuiToggleButton-root.Mui-selected": {
                  bgcolor: "rgba(78, 168, 255, 0.14)",
                },
              }}
            >
              <ToggleButton value="positive" sx={{ "&.Mui-selected": { color: "#00e676" } }}>
                Positive box
              </ToggleButton>
              <ToggleButton value="negative" sx={{ "&.Mui-selected": { color: "#ff5252" } }}>
                Negative box
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button size="small" variant="outlined" onClick={clearBoxes} disabled={disabled} sx={buttonSx}>
              Clear boxes
            </Button>
            <Button size="small" variant="contained" onClick={handleSegment} disabled={disabled} sx={buttonSx}>
              Segment
            </Button>
          </Stack>
        </>
      )}
      <Box
        ref={wrapRef}
        sx={{
          position: "relative",
          width: "100%",
          flex: 1,
          minHeight: 200,
          borderRadius: "8px",
          overflow: "hidden",
          border: "1px solid rgba(148, 163, 184, 0.16)",
          bgcolor: "#05080d",
          lineHeight: 0,
        }}
      >
        {imageSrc ? (
          <img
            src={imageSrc}
            alt="source"
            draggable={false}
            style={{ display: "block", width: "100%", height: "auto", userSelect: "none" }}
            onLoad={redraw}
          />
        ) : (
          <Box sx={{ p: 4, color: "#5f6e82", textAlign: "center", lineHeight: 1.5 }}>Upload an image to begin</Box>
        )}
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            zIndex: 1,
            pointerEvents: disabled ? "none" : "auto",
            cursor: disabled ? "default" : "crosshair",
            touchAction: "none",
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
          onLostPointerCapture={(e) => finishPointer(e)}
        />
      </Box>
    </Stack>
  );
}

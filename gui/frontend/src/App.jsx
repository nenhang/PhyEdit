import React, { useEffect, useRef, useState } from "react";
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Stack,
  Divider,
  Slider,
  TextField,
  Chip,
  FormControlLabel,
  IconButton,
  Switch,
  Tooltip,
} from "@mui/material";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import AutoAwesomeRoundedIcon from "@mui/icons-material/AutoAwesomeRounded";
import CenterFocusStrongRoundedIcon from "@mui/icons-material/CenterFocusStrongRounded";
import FileUploadRoundedIcon from "@mui/icons-material/FileUploadRounded";
import FolderOpenRoundedIcon from "@mui/icons-material/FolderOpenRounded";
import ImageRoundedIcon from "@mui/icons-material/ImageRounded";
import LayersRoundedIcon from "@mui/icons-material/LayersRounded";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import SaveRoundedIcon from "@mui/icons-material/SaveRounded";
import TuneRoundedIcon from "@mui/icons-material/TuneRounded";
import ViewInArRoundedIcon from "@mui/icons-material/ViewInArRounded";
import { Canvas } from "@react-three/fiber";
import axios from "axios";
import ScenePC from "./components/ScenePC";
import ImageSegmentPanel from "./components/ImageSegmentPanel";
import SimpleProjectionPreview from "./components/SimpleProjectionPreview";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const SLIDER_MAX = 1.0;
const Z_SLIDER_MAX_FALLBACK = 10.0;
const SLIDER_STEP = 0.005;
const ROT_MAX_DEG = 180;
const ROT_STEP_DEG = 1;
const DENSITY_MIN = 0.5;
const DENSITY_MAX = 12.0;
const DENSITY_STEP = 0.25;

function clampDelta(v, axis = "x", zMax = Z_SLIDER_MAX_FALLBACK) {
  if (!Number.isFinite(v)) return 0;
  const maxAbs = axis === "z" ? zMax : SLIDER_MAX;
  return Math.max(-maxAbs, Math.min(maxAbs, v));
}

function fmtDeltaStr(n) {
  if (!Number.isFinite(n)) return "0";
  let s = n.toFixed(6);
  s = s.replace(/\.?0+$/, "");
  if (s === "" || s === "-") return "0";
  return s;
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [pcUrl, setPcUrl] = useState(null);
  const [pcPreviewUrl, setPcPreviewUrl] = useState(null);
  const [pcCounts, setPcCounts] = useState({ full: 0, preview: 0 });
  const [basePreviewImageUrl, setBasePreviewImageUrl] = useState(null);
  const [previewImageUrl, setPreviewImageUrl] = useState(null);
  const [viewCommand, setViewCommand] = useState(null);
  const [imageKey, setImageKey] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [objects, setObjects] = useState([]);
  const [selectedObjectId, setSelectedObjectId] = useState(null);
  const [boxDrawMode, setBoxDrawMode] = useState("positive");
  const [deltaStr, setDeltaStr] = useState({ x: "0", y: "0", z: "0" });
  const [rotationStr, setRotationStr] = useState("0");
  const [rotationVStr, setRotationVStr] = useState("0");
  const [rotationHStr, setRotationHStr] = useState("0");
  const [renderedUrl, setRenderedUrl] = useState(null);
  const [generatedUrl, setGeneratedUrl] = useState(null);
  const [segBusy, setSegBusy] = useState(false);
  const [renderBusy, setRenderBusy] = useState(false);
  const [generateBusy, setGenerateBusy] = useState(false);
  const [saveBusy, setSaveBusy] = useState(false);
  const [projMeta, setProjMeta] = useState(null);
  const [renderDensity, setRenderDensity] = useState(2.5);
  const [autoDensity, setAutoDensity] = useState(true);
  const [holeFillAlgo, setHoleFillAlgo] = useState(false);
  const [holeFillColor, setHoleFillColor] = useState("#ffffff");
  const [additionalPrompt, setAdditionalPrompt] = useState("");
  const [randomGenerationSeed, setRandomGenerationSeed] = useState(true);
  const [generationSeed, setGenerationSeed] = useState(42);
  const [lastGenerationSeed, setLastGenerationSeed] = useState(null);
  const [generationSteps, setGenerationSteps] = useState(28);
  const [guidanceScale, setGuidanceScale] = useState(3.5);
  const [loadSceneDir, setLoadSceneDir] = useState("");
  const previewRenderSeqRef = useRef(0);
  const previewPayloadRef = useRef(null);
  const previewTimerRef = useRef(null);
  const previewInFlightRef = useRef(false);
  const previewQueuedRef = useRef(false);

  const imageSrc = imageKey ? `${API_BASE}${imageKey}` : null;

  const selectedObject = objects.find((o) => o.object_id === selectedObjectId) || null;
  const selectionActive = !!selectedObject;
  const maskUrl = selectedObject?.mask_url || null;
  const translation = selectedObject?.translation || { x: 0, y: 0, z: 0 };
  const rotationDeg = selectedObject?.rotation_deg || 0;
  const rotationVDeg = selectedObject?.rotation_v_deg || 0;
  const rotationHDeg = selectedObject?.rotation_h_deg || 0;
  const zSliderMax = (() => {
    const s = projMeta?.depthStats;
    if (!s) return Z_SLIDER_MAX_FALLBACK;
    const p95 = Number(s.p95);
    const p05 = Number(s.p05);
    const span = Math.max(0, p95 - p05);
    // True adaptive range: for large-depth scenes, allow much larger dz.
    // Keep a practical lower bound, but do not hard-cap at 10.
    const adaptive = Math.max(span * 0.9, Math.abs(p95) * 0.6, Z_SLIDER_MAX_FALLBACK);
    return Math.min(120.0, Math.max(5.0, Number.isFinite(adaptive) ? adaptive : Z_SLIDER_MAX_FALLBACK));
  })();

  const updateSelected = (patch) => {
    if (!selectedObjectId) return;
    setObjects((prev) => prev.map((o) => (o.object_id === selectedObjectId ? { ...o, ...patch } : o)));
  };

  const updateSelectedTranslation = (translationPatch) => {
    if (!selectedObjectId) return;
    const nextTranslation = {
      x: clampDelta(Number(translationPatch.x ?? translation.x), "x", zSliderMax),
      y: clampDelta(Number(translationPatch.y ?? translation.y), "y", zSliderMax),
      z: clampDelta(Number(translationPatch.z ?? translation.z), "z", zSliderMax),
    };
    updateSelected({ translation: nextTranslation });
    setDeltaStr({
      x: fmtDeltaStr(nextTranslation.x),
      y: fmtDeltaStr(nextTranslation.y),
      z: fmtDeltaStr(nextTranslation.z),
    });
  };

  const selectObject = (obj) => {
    setSelectedObjectId(obj?.object_id ?? null);
    setDeltaStr({
      x: fmtDeltaStr(obj?.translation?.x ?? 0),
      y: fmtDeltaStr(obj?.translation?.y ?? 0),
      z: fmtDeltaStr(obj?.translation?.z ?? 0),
    });
    setRotationStr(fmtDeltaStr(obj?.rotation_deg ?? 0));
    setRotationVStr(fmtDeltaStr(obj?.rotation_v_deg ?? 0));
    setRotationHStr(fmtDeltaStr(obj?.rotation_h_deg ?? 0));
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setIsLoading(true);
    setObjects([]);
    setSelectedObjectId(null);
    setRenderedUrl(null);
    setGeneratedUrl(null);
    setProjMeta(null);
    setPcPreviewUrl(null);
    setPcCounts({ full: 0, preview: 0 });
    setBasePreviewImageUrl(null);
    setPreviewImageUrl(null);
    setDeltaStr({ x: "0", y: "0", z: "0" });
    setRotationStr("0");
    setRotationVStr("0");
    setRotationHStr("0");
    const fd = new FormData();
    fd.append("image", file);
    try {
      const res = await axios.post(`${API_BASE}/predict-depth`, fd);
      setSessionId(res.data.session_id);
      setPcUrl(`${API_BASE}${res.data.pc_url}`);
      setPcPreviewUrl(`${API_BASE}${res.data.pc_preview_url || res.data.pc_url}`);
      const basePreviewUrl = `${API_BASE}${res.data.preview_image_url || res.data.image_key}`;
      setBasePreviewImageUrl(basePreviewUrl);
      setPreviewImageUrl(basePreviewUrl);
      setPcCounts({
        full: Number(res.data.pc_point_count || 0),
        preview: Number(res.data.pc_preview_point_count || res.data.pc_point_count || 0),
      });
      setImageKey(res.data.image_key);
      setProjMeta({
        intrinsics: res.data.intrinsics,
        imageWidth: res.data.image_width,
        imageHeight: res.data.image_height,
        depthStats: res.data.depth_stats || null,
      });
    } catch (err) {
      console.error(err);
      alert("Reconstruction failed. Check the backend and DA3 model configuration.");
    } finally {
      setIsLoading(false);
    }
  };

  const loadScene = async (sceneDirOverride) => {
    const sceneDir = (typeof sceneDirOverride === "string" ? sceneDirOverride : loadSceneDir).trim();
    if (!sceneDir) return;
    setIsLoading(true);
    setBasePreviewImageUrl(null);
    setPreviewImageUrl(null);
    try {
      const res = await axios.post(`${API_BASE}/load-scene`, {
        scene_dir: sceneDir,
        recompute_depth: true,
      });
      setSessionId(res.data.session_id);
      setPcUrl(`${API_BASE}${res.data.pc_url}`);
      setPcPreviewUrl(`${API_BASE}${res.data.pc_preview_url || res.data.pc_url}`);
      const basePreviewUrl = `${API_BASE}${res.data.preview_image_url || res.data.image_key}`;
      setBasePreviewImageUrl(basePreviewUrl);
      setPreviewImageUrl(basePreviewUrl);
      setPcCounts({
        full: Number(res.data.pc_point_count || 0),
        preview: Number(res.data.pc_preview_point_count || res.data.pc_point_count || 0),
      });
      setImageKey(res.data.image_key);
      setProjMeta({
        intrinsics: res.data.intrinsics,
        imageWidth: res.data.image_width,
        imageHeight: res.data.image_height,
        depthStats: res.data.depth_stats || null,
      });
      const objs = (res.data.objects || []).map((obj, index) => ({
        ...obj,
        text: String(obj.text || "").trim() || `object ${index + 1}`,
      }));
      setObjects(objs);
      setRenderedUrl(null);
      setGeneratedUrl(null);
      selectObject(objs[0] || null);
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail;
      alert(typeof msg === "string" ? msg : "Failed to load the saved scene.");
    } finally {
      setIsLoading(false);
    }
  };

  const runSegment = async (payload) => {
    if (!sessionId) return;
    const hasText = payload.text && String(payload.text).trim();
    const hasBox = payload.input_boxes?.[0]?.length > 0;
    if (!hasText && !hasBox) {
      alert("Enter a text prompt, draw a box, or use both.");
      return;
    }
    setSegBusy(true);
    try {
      const res = await axios.post(`${API_BASE}/segment`, {
        session_id: sessionId,
        ...payload,
      });
      if (res.data.mask_url) {
        const objectId = crypto.randomUUID().slice(0, 8);
        const objectName = String(payload.text || "").trim() || `object ${objects.length + 1}`;
        const obj = {
          object_id: objectId,
          mask_url: res.data.mask_url,
          text: objectName,
          translation: { x: 0, y: 0, z: 0 },
          rotation_deg: 0,
          rotation_v_deg: 0,
          rotation_h_deg: 0,
        };
        setObjects((prev) => [...prev, obj]);
        setSelectedObjectId(objectId);
        setDeltaStr({ x: "0", y: "0", z: "0" });
        setRotationStr("0");
        setRotationVStr("0");
        setRotationHStr("0");
      } else {
        alert(res.data.message || "No mask was returned.");
      }
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail;
      alert(typeof msg === "string" ? msg : "Segmentation failed. Check the SAM 3 configuration.");
    } finally {
      setSegBusy(false);
    }
  };

  const clearSelection = async () => {
    setSelectedObjectId(null);
    setDeltaStr({ x: "0", y: "0", z: "0" });
    setRotationStr("0");
    setRotationVStr("0");
    setRotationHStr("0");
  };

  const resetTranslation = () => {
    updateSelected({
      translation: { x: 0, y: 0, z: 0 },
      rotation_deg: 0,
      rotation_v_deg: 0,
      rotation_h_deg: 0,
    });
    setDeltaStr({ x: "0", y: "0", z: "0" });
    setRotationStr("0");
    setRotationVStr("0");
    setRotationHStr("0");
  };

  const renderMoved = async () => {
    if (!sessionId) return;
    if (objects.length === 0) {
      alert("Segment or import at least one object first.");
      return;
    }
    setRenderBusy(true);
    setGeneratedUrl(null);
    try {
      const payload = {
        session_id: sessionId,
        hole_fill_algo: holeFillAlgo,
        hole_fill_rgb: [
          parseInt(holeFillColor.slice(1, 3), 16),
          parseInt(holeFillColor.slice(3, 5), 16),
          parseInt(holeFillColor.slice(5, 7), 16),
        ],
        objects: objects.map((o) => ({
          object_id: o.object_id,
          mask_url: o.mask_url,
          dx: o.translation.x,
          dy: o.translation.y,
          dz: o.translation.z,
          rot_deg: o.rotation_deg,
          rot_v_deg: o.rotation_v_deg ?? 0,
          rot_h_deg: o.rotation_h_deg ?? 0,
          text: o.text,
        })),
      };
      if (!autoDensity) payload.density = renderDensity;
      const res = await axios.post(`${API_BASE}/render-multi`, payload);
      setRenderedUrl(`${API_BASE}${res.data.image_url}`);
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail;
      alert(typeof msg === "string" ? msg : "Rendering failed.");
    } finally {
      setRenderBusy(false);
    }
  };

  const generateFinal = async () => {
    if (!sessionId || objects.length === 0) return;
    setGenerateBusy(true);
    try {
      const payload = {
        session_id: sessionId,
        hole_fill_algo: holeFillAlgo,
        hole_fill_rgb: [
          parseInt(holeFillColor.slice(1, 3), 16),
          parseInt(holeFillColor.slice(3, 5), 16),
          parseInt(holeFillColor.slice(5, 7), 16),
        ],
        objects: objects.map((o) => ({
          object_id: o.object_id,
          mask_url: o.mask_url,
          dx: o.translation.x,
          dy: o.translation.y,
          dz: o.translation.z,
          rot_deg: o.rotation_deg,
          rot_v_deg: o.rotation_v_deg ?? 0,
          rot_h_deg: o.rotation_h_deg ?? 0,
          text: o.text,
        })),
        additional_prompt: additionalPrompt.trim() || null,
        num_inference_steps: Math.trunc(generationSteps),
        guidance_scale: guidanceScale,
      };
      if (!randomGenerationSeed) payload.seed = Math.trunc(generationSeed);
      if (!autoDensity) payload.density = renderDensity;
      const res = await axios.post(`${API_BASE}/generate`, payload);
      setRenderedUrl(`${API_BASE}${res.data.moved_image_url}`);
      setGeneratedUrl(`${API_BASE}${res.data.image_url}`);
      setLastGenerationSeed(Number.isInteger(res.data.seed) ? res.data.seed : null);
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail;
      alert(typeof msg === "string" ? msg : "PhyEdit generation failed.");
    } finally {
      setGenerateBusy(false);
    }
  };

  useEffect(() => {
    return () => {
      if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
    };
  }, []);

  useEffect(() => {
    setRenderedUrl(null);
    setGeneratedUrl(null);
  }, [sessionId, objects, holeFillAlgo, holeFillColor, autoDensity, renderDensity]);

  useEffect(() => {
    setGeneratedUrl(null);
  }, [additionalPrompt, randomGenerationSeed, generationSeed, generationSteps, guidanceScale]);

  useEffect(() => {
    if (!sessionId || objects.length === 0) {
      previewRenderSeqRef.current += 1;
      previewPayloadRef.current = null;
      previewQueuedRef.current = false;
      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
        previewTimerRef.current = null;
      }
      setPreviewImageUrl(basePreviewImageUrl);
      return;
    }

    const payload = {
      session_id: sessionId,
      hole_fill_algo: holeFillAlgo,
      hole_fill_rgb: [
        parseInt(holeFillColor.slice(1, 3), 16),
        parseInt(holeFillColor.slice(3, 5), 16),
        parseInt(holeFillColor.slice(5, 7), 16),
      ],
      objects: objects.map((o) => ({
        object_id: o.object_id,
        mask_url: o.mask_url,
        dx: o.translation.x,
        dy: o.translation.y,
        dz: o.translation.z,
        rot_deg: o.rotation_deg,
        rot_v_deg: o.rotation_v_deg ?? 0,
        rot_h_deg: o.rotation_h_deg ?? 0,
        text: o.text,
      })),
    };
    if (!autoDensity) payload.density = renderDensity;
    previewPayloadRef.current = payload;
    previewQueuedRef.current = true;

    const flushPreview = async () => {
      previewTimerRef.current = null;
      if (previewInFlightRef.current) return;
      const nextPayload = previewPayloadRef.current;
      if (!nextPayload) return;
      const seq = previewRenderSeqRef.current + 1;
      previewRenderSeqRef.current = seq;
      previewQueuedRef.current = false;
      previewInFlightRef.current = true;
      try {
        const res = await axios.post(`${API_BASE}/preview-render`, nextPayload);
        if (
          previewRenderSeqRef.current !== seq ||
          !res.data.image_url ||
          previewPayloadRef.current !== nextPayload
        ) {
          return;
        }
        setPreviewImageUrl(`${API_BASE}${res.data.image_url}?t=${Date.now()}`);
      } catch (err) {
        if (err?.response?.status !== 404) {
          console.warn("preview-render failed", err);
        }
      } finally {
        previewInFlightRef.current = false;
        if (previewQueuedRef.current && !previewTimerRef.current) {
          previewTimerRef.current = setTimeout(flushPreview, 80);
        }
      }
    };

    if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
    previewTimerRef.current = setTimeout(flushPreview, 220);
  }, [sessionId, objects, basePreviewImageUrl, holeFillAlgo, holeFillColor, autoDensity, renderDensity]);

  const saveScene = async () => {
    if (!sessionId) return;
    if (objects.length === 0) {
      alert("There are no objects to save.");
      return;
    }
    setSaveBusy(true);
    try {
      const res = await axios.post(`${API_BASE}/save-scene`, {
        session_id: sessionId,
        rendered_image_url: renderedUrl ? renderedUrl.replace(API_BASE, "") : null,
        generated_image_url: generatedUrl ? generatedUrl.replace(API_BASE, "") : null,
        objects: objects.map((o) => ({
          object_id: o.object_id,
          mask_url: o.mask_url,
          dx: o.translation.x,
          dy: o.translation.y,
          dz: o.translation.z,
          rot_deg: o.rotation_deg,
          rot_v_deg: o.rotation_v_deg ?? 0,
          rot_h_deg: o.rotation_h_deg ?? 0,
          text: o.text,
        })),
      });
      alert(`Saved to: ${res.data.save_dir}`);
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail;
      alert(typeof msg === "string" ? msg : "Failed to save the scene.");
    } finally {
      setSaveBusy(false);
    }
  };

  const uploadMasksObjects = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length === 0 || !sessionId) return;
    const fd = new FormData();
    fd.append("session_id", sessionId);
    for (const f of files) fd.append("masks", f);
    setSegBusy(true);
    try {
      const res = await axios.post(`${API_BASE}/upload-masks`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const maskUrls = res.data.mask_urls || [];
      if (maskUrls.length === 0) throw new Error("no mask_urls returned");

      const newObjects = maskUrls.map((maskUrl, index) => {
        const objectId = crypto.randomUUID().slice(0, 8);
        return {
          object_id: objectId,
          mask_url: maskUrl,
          text: `object ${objects.length + index + 1}`,
          translation: { x: 0, y: 0, z: 0 },
          rotation_deg: 0,
          rotation_v_deg: 0,
          rotation_h_deg: 0,
        };
      });

      setObjects((prev) => [...prev, ...newObjects]);
      const last = newObjects[newObjects.length - 1];
      selectObject(last);
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.detail;
      alert(typeof msg === "string" ? msg : "Failed to upload masks.");
    } finally {
      setSegBusy(false);
    }
  };

  const deleteObjectById = (objectId) => {
    const remain = objects.filter((o) => o.object_id !== objectId);
    setObjects(remain);
    if (selectedObjectId === objectId) {
      selectObject(remain[0] || null);
    }
  };

  const sliderSx = {
    color: "#4ea8ff",
    height: 4,
    "& .MuiSlider-thumb": {
      width: 14,
      height: 14,
      bgcolor: "#d8ecff",
      border: "2px solid #4ea8ff",
      boxShadow: "0 0 0 4px rgba(78, 168, 255, 0.1)",
    },
    "& .MuiSlider-track": { bgcolor: "#4ea8ff", border: 0 },
    "& .MuiSlider-rail": { bgcolor: "rgba(148,163,184,0.22)" },
  };

  const panelSx = {
    bgcolor: "#10151d",
    color: "#e8eef7",
    borderRadius: "8px",
    border: "1px solid rgba(148, 163, 184, 0.16)",
    boxShadow: "none",
  };
  const sectionSx = {
    py: 1.35,
    borderTop: "1px solid rgba(148, 163, 184, 0.12)",
    "&:first-of-type": { borderTop: 0, pt: 0 },
  };
  const sectionTitleSx = {
    color: "#c8d3e2",
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: 0,
    mb: 1,
  };
  const captionSx = {
    color: "#8b98aa",
    fontSize: 12,
  };
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
  const numberFieldSx = {
    ...fieldSx,
    width: 86,
    flexShrink: 0,
    "& .MuiInputBase-input": {
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      py: 0.65,
      color: "#e8eef7",
      textAlign: "right",
    },
  };

  const selectedLabel = selectedObject
    ? selectedObject.text || selectedObject.object_id
    : "No selection";
  const browserPcUrl = pcPreviewUrl || pcUrl;
  const focusPointCloud = (mode) => {
    setViewCommand({ mode, token: Date.now() });
  };

  const renderProjectionPreview = (
    <SimpleProjectionPreview
      pcUrl={browserPcUrl}
      previewImageUrl={previewImageUrl}
      objects={objects}
      selectedObjectId={selectedObjectId}
      intrinsics={projMeta?.intrinsics}
      imageWidth={projMeta?.imageWidth}
      imageHeight={projMeta?.imageHeight}
      apiBase={API_BASE}
    />
  );

  const renderAxisControl = (axis) => (
    <Box key={axis}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.15 }}>
        <Typography variant="caption" sx={{ color: "#a8b4c5", fontWeight: 700 }}>
          {`Delta ${axis}`}
        </Typography>
        <Typography variant="caption" sx={captionSx}>
          {axis === "z" ? `Range +/-${fmtDeltaStr(zSliderMax)}` : `Range +/-${SLIDER_MAX}`}
        </Typography>
      </Stack>
      <Stack direction="row" spacing={1} alignItems="center">
        <Slider
          size="small"
          min={axis === "z" ? -zSliderMax : -SLIDER_MAX}
          max={axis === "z" ? zSliderMax : SLIDER_MAX}
          step={SLIDER_STEP}
          value={translation[axis]}
          onChange={(_, v) => {
            updateSelected({ translation: { ...translation, [axis]: v } });
            setDeltaStr((s) => ({ ...s, [axis]: fmtDeltaStr(v) }));
          }}
          disabled={!selectionActive || !maskUrl}
          sx={{ ...sliderSx, flex: 1, minWidth: 0 }}
        />
        <TextField
          size="small"
          type="text"
          inputMode="decimal"
          disabled={!selectionActive || !maskUrl}
          value={deltaStr[axis]}
          onChange={(e) => {
            const raw = e.target.value;
            setDeltaStr((s) => ({ ...s, [axis]: raw }));
            if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
            const v = parseFloat(raw);
            if (!Number.isFinite(v)) return;
            updateSelected({ translation: { ...translation, [axis]: clampDelta(v, axis, zSliderMax) } });
          }}
          onBlur={() => {
            const v = parseFloat(deltaStr[axis]);
            if (!Number.isFinite(v)) {
              setDeltaStr((s) => ({ ...s, [axis]: fmtDeltaStr(translation[axis]) }));
              return;
            }
            const c = clampDelta(v, axis, zSliderMax);
            updateSelected({ translation: { ...translation, [axis]: c } });
            setDeltaStr((s) => ({ ...s, [axis]: fmtDeltaStr(c) }));
          }}
          inputProps={{
            step: SLIDER_STEP,
            "aria-label": `delta-${axis}`,
          }}
          sx={numberFieldSx}
        />
      </Stack>
    </Box>
  );

  const renderRotationControl = (label, value, strValue, setStrValue, patchKey) => (
    <Box>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 0.15 }}>
        <Typography variant="caption" sx={{ color: "#a8b4c5", fontWeight: 700 }}>
          {label}
        </Typography>
        <Typography variant="caption" sx={captionSx}>
          deg
        </Typography>
      </Stack>
      <Stack direction="row" spacing={1} alignItems="center">
        <Slider
          size="small"
          min={-ROT_MAX_DEG}
          max={ROT_MAX_DEG}
          step={ROT_STEP_DEG}
          value={value}
          onChange={(_, v) => {
            updateSelected({ [patchKey]: v });
            setStrValue(fmtDeltaStr(v));
          }}
          disabled={!selectionActive || !maskUrl}
          sx={{ ...sliderSx, flex: 1, minWidth: 0 }}
        />
        <TextField
          size="small"
          type="text"
          inputMode="decimal"
          disabled={!selectionActive || !maskUrl}
          value={strValue}
          onChange={(e) => {
            const raw = e.target.value;
            setStrValue(raw);
            if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
            const v = parseFloat(raw);
            if (!Number.isFinite(v)) return;
            const c = Math.max(-ROT_MAX_DEG, Math.min(ROT_MAX_DEG, v));
            updateSelected({ [patchKey]: c });
          }}
          onBlur={() => {
            const v = parseFloat(strValue);
            const c = Number.isFinite(v) ? Math.max(-ROT_MAX_DEG, Math.min(ROT_MAX_DEG, v)) : value;
            updateSelected({ [patchKey]: c });
            setStrValue(fmtDeltaStr(c));
          }}
          sx={numberFieldSx}
        />
      </Stack>
    </Box>
  );

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        width: "100vw",
        bgcolor: "#0b1017",
        color: "#e8eef7",
        overflow: "hidden",
      }}
    >
      <Box
        component="header"
        sx={{
          height: 58,
          flexShrink: 0,
          px: 2,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 2,
          bgcolor: "#0d131b",
          borderBottom: "1px solid rgba(148, 163, 184, 0.14)",
        }}
      >
        <Stack direction="row" spacing={1.25} alignItems="center" sx={{ minWidth: 0 }}>
          <Box
            sx={{
              width: 34,
              height: 34,
              borderRadius: "8px",
              display: "grid",
              placeItems: "center",
              color: "#9bd3ff",
              bgcolor: "rgba(78, 168, 255, 0.12)",
              border: "1px solid rgba(78, 168, 255, 0.25)",
              flexShrink: 0,
            }}
          >
            <ViewInArRoundedIcon fontSize="small" />
          </Box>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 800, lineHeight: 1.15 }}>
              PhyEdit Editor
            </Typography>
            <Typography variant="caption" sx={{ color: "#7d8ca1", display: { xs: "none", sm: "block" } }}>
              3D-aware object manipulation and image generation
            </Typography>
          </Box>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: "wrap", justifyContent: "flex-end" }}>
          <Chip
            size="small"
            variant="outlined"
            label={sessionId ? `Session ${sessionId.slice(0, 8)}` : "No scene loaded"}
            sx={{ color: "#c8d3e2", borderColor: "rgba(148, 163, 184, 0.28)" }}
          />
          <Chip
            size="small"
            variant="outlined"
            label={`${objects.length} object${objects.length === 1 ? "" : "s"}`}
            sx={{ color: objects.length ? "#9ee8c5" : "#8b98aa", borderColor: "rgba(148, 163, 184, 0.28)" }}
          />
          <Chip
            size="small"
            label={selectedObject ? "Editing" : "No selection"}
            sx={{
              color: selectedObject ? "#0b1017" : "#8b98aa",
              bgcolor: selectedObject ? "#9bd3ff" : "rgba(148, 163, 184, 0.08)",
            }}
          />
        </Stack>
      </Box>

      <Box
        component="main"
        sx={{
          flex: 1,
          minHeight: 0,
          p: 1.5,
          display: "grid",
          gridTemplateColumns: {
            xs: "minmax(0, 1fr)",
            lg: "340px minmax(360px, 1fr) minmax(420px, 1.15fr)",
          },
          gridTemplateRows: {
            xs: "max-content max-content max-content",
            lg: "minmax(0, 1fr)",
          },
          alignContent: "start",
          gap: 1.5,
          overflow: { xs: "auto", lg: "hidden" },
        }}
      >
        <Paper
          sx={{
            ...panelSx,
            p: 1.5,
            display: "flex",
            flexDirection: "column",
            minHeight: { xs: "max-content", lg: 0 },
            overflowY: { xs: "visible", lg: "auto" },
          }}
        >
          <Box sx={sectionSx}>
            <Typography sx={sectionTitleSx}>Input</Typography>
            <Button
              variant="contained"
              component="label"
              fullWidth
              disabled={isLoading}
              startIcon={!isLoading ? <FileUploadRoundedIcon /> : null}
              sx={{ py: 1.05, fontWeight: 800, borderRadius: "6px" }}
            >
              {isLoading ? <CircularProgress size={20} color="inherit" /> : "Upload and reconstruct"}
              <input type="file" accept="image/*" hidden onChange={handleUpload} />
            </Button>

            <Stack direction="row" spacing={1} sx={{ mt: 1.15 }}>
              <TextField
                size="small"
                label="Saved scene directory"
                value={loadSceneDir}
                onChange={(e) => setLoadSceneDir(e.target.value)}
                disabled={isLoading}
                fullWidth
                placeholder="20260802_120000_ab12cd34"
                sx={fieldSx}
              />
              <Button
                variant="outlined"
                onClick={() => loadScene()}
                disabled={isLoading || !loadSceneDir.trim()}
                startIcon={<FolderOpenRoundedIcon />}
                sx={{ borderRadius: "6px", minWidth: 86 }}
              >
                Restore
              </Button>
            </Stack>
          </Box>

          <Box sx={sectionSx}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
              <Typography sx={{ ...sectionTitleSx, mb: 0 }}>Object pose</Typography>
              <Chip
                size="small"
                label={selectedLabel}
                sx={{
                  maxWidth: 160,
                  color: selectedObject ? "#9bd3ff" : "#7d8ca1",
                  bgcolor: "rgba(148, 163, 184, 0.08)",
                  borderRadius: "6px",
                  "& .MuiChip-label": {
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  },
                }}
              />
            </Stack>

            <Stack spacing={1.05}>
              <TextField
                size="small"
                label="Object name"
                value={selectedObject?.text || ""}
                onChange={(e) => updateSelected({ text: e.target.value })}
                disabled={!selectedObject}
                fullWidth
                placeholder="e.g. purple toy car"
                inputProps={{ maxLength: 120 }}
                sx={fieldSx}
              />
              <Divider sx={{ borderColor: "rgba(148, 163, 184, 0.12)", my: 0.2 }} />
              {["x", "y", "z"].map(renderAxisControl)}
              <Divider sx={{ borderColor: "rgba(148, 163, 184, 0.12)", my: 0.2 }} />
              {renderRotationControl("In-plane rotation", rotationDeg, rotationStr, setRotationStr, "rotation_deg")}
              {renderRotationControl("Vertical-axis rotation", rotationVDeg, rotationVStr, setRotationVStr, "rotation_v_deg")}
              {renderRotationControl("Horizontal-axis rotation", rotationHDeg, rotationHStr, setRotationHStr, "rotation_h_deg")}
            </Stack>

            <Stack direction="row" sx={{ mt: 1.35, flexWrap: "wrap", gap: 0.75 }}>
              <Button
                size="small"
                variant="outlined"
                onClick={resetTranslation}
                disabled={!maskUrl}
                startIcon={<RestartAltRoundedIcon />}
                sx={{ borderRadius: "6px" }}
              >
                Reset
              </Button>
              <Button
                size="small"
                variant="outlined"
                component="label"
                disabled={!sessionId || segBusy}
                startIcon={<LayersRoundedIcon />}
                sx={{ borderRadius: "6px" }}
              >
                {segBusy ? "Processing" : "Import masks"}
                <input type="file" accept="image/*" multiple hidden onChange={uploadMasksObjects} />
              </Button>
              <Button
                size="small"
                color="warning"
                variant="outlined"
                onClick={clearSelection}
                disabled={!sessionId}
                sx={{ borderRadius: "6px" }}
              >
                Clear selection
              </Button>
              <Button
                size="small"
                variant="outlined"
                onClick={saveScene}
                disabled={saveBusy || !sessionId}
                startIcon={<SaveRoundedIcon />}
                sx={{ borderRadius: "6px" }}
              >
                {saveBusy ? "Saving" : "Save"}
              </Button>
            </Stack>
          </Box>

          <Box sx={sectionSx}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
              <Typography sx={{ ...sectionTitleSx, mb: 0 }}>Objects</Typography>
              <Typography variant="caption" sx={captionSx}>
                Click to select
              </Typography>
            </Stack>
            {objects.length === 0 ? (
              <Box
                sx={{
                  p: 1.25,
                  border: "1px dashed rgba(148, 163, 184, 0.22)",
                  borderRadius: "8px",
                  color: "#74849a",
                  textAlign: "center",
                  fontSize: 13,
                }}
              >
                No objects
              </Box>
            ) : (
              <Stack spacing={0.75}>
                {objects.map((o, idx) => {
                  const active = o.object_id === selectedObjectId;
                  return (
                    <Box
                      key={o.object_id}
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 0.75,
                        p: 0.75,
                        borderRadius: "8px",
                        border: `1px solid ${active ? "rgba(78,168,255,0.5)" : "rgba(148,163,184,0.14)"}`,
                        bgcolor: active ? "rgba(78,168,255,0.12)" : "rgba(255,255,255,0.025)",
                      }}
                    >
                      <Button
                        size="small"
                        variant="text"
                        onClick={() => selectObject(o)}
                        sx={{
                          minWidth: 0,
                          flex: 1,
                          p: 0,
                          justifyContent: "flex-start",
                          textTransform: "none",
                          color: active ? "#d8ecff" : "#c8d3e2",
                        }}
                      >
                        <Box sx={{ minWidth: 0, textAlign: "left" }}>
                          <Typography sx={{ fontSize: 13, fontWeight: 800 }} noWrap>
                            {String(o.text || "").trim() || `object ${idx + 1}`}
                          </Typography>
                          <Typography variant="caption" sx={{ color: "#8b98aa", display: "block" }} noWrap>
                            {`Object ${idx + 1}`}
                          </Typography>
                        </Box>
                      </Button>
                      <Tooltip title="Delete object">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => deleteObjectById(o.object_id)}
                          sx={{ borderRadius: "6px", flexShrink: 0 }}
                        >
                          <DeleteOutlineRoundedIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  );
                })}
              </Stack>
            )}
          </Box>

          <Box sx={sectionSx}>
            <Stack direction="row" spacing={0.75} alignItems="center" sx={{ mb: 1 }}>
              <TuneRoundedIcon fontSize="small" sx={{ color: "#8cc8ff" }} />
              <Typography sx={{ ...sectionTitleSx, mb: 0 }}>Render settings</Typography>
            </Stack>

            <Button
              variant="contained"
              fullWidth
              startIcon={!renderBusy ? <PlayArrowRoundedIcon /> : null}
              sx={{ bgcolor: "#1769aa", borderRadius: "6px", py: 1, fontWeight: 800 }}
              disabled={!sessionId || objects.length === 0 || renderBusy}
              onClick={renderMoved}
            >
              {renderBusy ? <CircularProgress size={20} color="inherit" /> : "Render moved image"}
            </Button>

            <Stack spacing={0.75} sx={{ mt: 1.15 }}>
              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={autoDensity}
                    onChange={(e) => setAutoDensity(e.target.checked)}
                  />
                }
                label={<Typography variant="caption" sx={{ color: "#c8d3e2" }}>Automatic density</Typography>}
                sx={{ m: 0 }}
              />
              <Stack direction="row" spacing={1} alignItems="center">
                <Slider
                  size="small"
                  min={DENSITY_MIN}
                  max={DENSITY_MAX}
                  step={DENSITY_STEP}
                  value={renderDensity}
                  onChange={(_, v) => setRenderDensity(Number(v))}
                  disabled={autoDensity}
                  sx={{ ...sliderSx, flex: 1, minWidth: 0 }}
                />
                <TextField
                  size="small"
                  type="text"
                  inputMode="decimal"
                  value={fmtDeltaStr(renderDensity)}
                  disabled={autoDensity}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
                    const v = parseFloat(raw);
                    if (!Number.isFinite(v)) return;
                    const c = Math.max(DENSITY_MIN, Math.min(DENSITY_MAX, v));
                    setRenderDensity(c);
                  }}
                  sx={numberFieldSx}
                />
              </Stack>

              <FormControlLabel
                control={
                  <Switch
                    size="small"
                    checked={holeFillAlgo}
                    onChange={(e) => setHoleFillAlgo(e.target.checked)}
                  />
                }
                label={<Typography variant="caption" sx={{ color: "#c8d3e2" }}>Inpaint vacated regions</Typography>}
                sx={{ m: 0 }}
              />
              {!holeFillAlgo && (
                <Stack direction="row" spacing={1} alignItems="center">
                  <Typography variant="caption" sx={captionSx}>
                    Fill color
                  </Typography>
                  <TextField
                    size="small"
                    type="color"
                    value={holeFillColor}
                    onChange={(e) => setHoleFillColor(e.target.value)}
                    sx={{
                      ...fieldSx,
                      width: 54,
                      "& .MuiInputBase-input": { p: 0.25, height: 30 },
                    }}
                  />
                  <Typography variant="caption" sx={{ color: "#a8b4c5", fontFamily: "monospace" }}>
                    {holeFillColor}
                  </Typography>
                </Stack>
              )}
            </Stack>
          </Box>

          <Box sx={sectionSx}>
            <Stack direction="row" spacing={0.75} alignItems="center" sx={{ mb: 1 }}>
              <AutoAwesomeRoundedIcon fontSize="small" sx={{ color: "#9ee8c5" }} />
              <Typography sx={{ ...sectionTitleSx, mb: 0 }}>PhyEdit generation</Typography>
            </Stack>

            <TextField
              size="small"
              label="Additional instruction"
              value={additionalPrompt}
              onChange={(e) => setAdditionalPrompt(e.target.value)}
              fullWidth
              multiline
              minRows={2}
              sx={fieldSx}
            />

            <FormControlLabel
              control={
                <Switch
                  checked={randomGenerationSeed}
                  onChange={(e) => setRandomGenerationSeed(e.target.checked)}
                  size="small"
                />
              }
              label="Random seed"
              sx={{ mt: 0.8, color: "#c8d3e2" }}
            />

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                gap: 0.75,
                mt: 1,
              }}
            >
              <TextField
                size="small"
                type="number"
                label="Seed"
                value={generationSeed}
                disabled={randomGenerationSeed}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (Number.isFinite(value)) {
                    setGenerationSeed(Math.max(0, Math.min(2_147_483_647, Math.trunc(value))));
                  }
                }}
                slotProps={{ htmlInput: { min: 0, max: 2_147_483_647, step: 1 } }}
                sx={fieldSx}
              />
              <TextField
                size="small"
                type="number"
                label="Steps"
                value={generationSteps}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (Number.isFinite(value)) setGenerationSteps(Math.max(1, Math.min(100, Math.trunc(value))));
                }}
                slotProps={{ htmlInput: { min: 1, max: 100 } }}
                sx={fieldSx}
              />
              <TextField
                size="small"
                type="number"
                label="Guidance"
                value={guidanceScale}
                onChange={(e) => {
                  const value = Number(e.target.value);
                  if (Number.isFinite(value)) setGuidanceScale(Math.max(0, Math.min(20, value)));
                }}
                slotProps={{ htmlInput: { min: 0, max: 20, step: 0.1 } }}
                sx={fieldSx}
              />
            </Box>

            <Button
              variant="contained"
              fullWidth
              startIcon={!generateBusy ? <AutoAwesomeRoundedIcon /> : null}
              sx={{ bgcolor: "#16785b", borderRadius: "6px", py: 1, mt: 1, fontWeight: 800 }}
              disabled={!sessionId || objects.length === 0 || generateBusy}
              onClick={generateFinal}
            >
              {generateBusy ? (
                <Stack component="span" direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={18} color="inherit" />
                  <span>Generating...</span>
                </Stack>
              ) : (
                "Generate with PhyEdit"
              )}
            </Button>
          </Box>
        </Paper>

        <Paper
          sx={{
            ...panelSx,
            p: 1.5,
            display: "flex",
            flexDirection: "column",
            minHeight: { xs: 560, lg: 0 },
            overflow: "hidden",
          }}
        >
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.2 }}>
            <Stack direction="row" spacing={0.75} alignItems="center">
              <ImageRoundedIcon fontSize="small" sx={{ color: "#8cc8ff" }} />
              <Box>
                <Typography sx={{ fontWeight: 800, lineHeight: 1.2 }}>Image and masks</Typography>
                <Typography variant="caption" sx={captionSx}>
                  {projMeta?.imageWidth && projMeta?.imageHeight
                    ? `${projMeta.imageWidth} x ${projMeta.imageHeight}`
                    : "Waiting for input"}
                </Typography>
              </Box>
            </Stack>
            <Chip
              size="small"
              label={boxDrawMode === "positive" ? "Positive box" : "Negative box"}
              sx={{ bgcolor: "rgba(148, 163, 184, 0.08)", color: "#c8d3e2", borderRadius: "6px" }}
            />
          </Stack>
          <Box sx={{ flex: 1, minHeight: 0, overflow: "auto" }}>
            <ImageSegmentPanel
              key={sessionId || "none"}
              imageSrc={imageSrc}
              disabled={!sessionId || isLoading || segBusy}
              onRunSegment={runSegment}
              boxDrawMode={boxDrawMode}
              onBoxDrawModeChange={setBoxDrawMode}
              objects={objects}
              selectedObjectId={selectedObjectId}
              apiBase={API_BASE}
            />
          </Box>
        </Paper>

        <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, gap: 1.5 }}>
          <Paper
            sx={{
              ...panelSx,
              p: 1.5,
              flex: "1 1 54%",
              minHeight: { xs: 420, lg: 0 },
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.2 }}>
              <Stack direction="row" spacing={0.75} alignItems="center">
                <ViewInArRoundedIcon fontSize="small" sx={{ color: "#8cc8ff" }} />
                <Box>
                  <Typography sx={{ fontWeight: 800, lineHeight: 1.2 }}>Point cloud</Typography>
                  <Typography variant="caption" sx={captionSx}>
                    {browserPcUrl
                      ? `Browser preview: ${pcCounts.preview ? pcCounts.preview.toLocaleString() : "sampled"} points`
                      : "Waiting for reconstruction"}
                  </Typography>
                </Box>
              </Stack>
              <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexWrap: "wrap", justifyContent: "flex-end" }}>
                <Button
                  size="small"
                  variant="outlined"
                  disabled={!browserPcUrl}
                  onClick={() => focusPointCloud("scene")}
                  startIcon={<ViewInArRoundedIcon />}
                  sx={{ borderRadius: "6px", minWidth: 74 }}
                >
                  Scene
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  disabled={!browserPcUrl || !selectedObject}
                  onClick={() => focusPointCloud("selected")}
                  startIcon={<CenterFocusStrongRoundedIcon />}
                  sx={{ borderRadius: "6px", minWidth: 74 }}
                >
                  Selection
                </Button>
                <Chip
                  size="small"
                  label={pcUrl ? "Loaded" : "Empty"}
                  sx={{
                    bgcolor: pcUrl ? "rgba(64, 211, 143, 0.14)" : "rgba(148, 163, 184, 0.08)",
                    color: pcUrl ? "#9ee8c5" : "#8b98aa",
                    borderRadius: "6px",
                  }}
                />
              </Stack>
            </Stack>

            <Box sx={{ flex: 1, minHeight: 0, position: "relative", borderRadius: "8px", overflow: "hidden", bgcolor: "#05080d" }}>
              {isLoading && (
                <Box
                  sx={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    zIndex: 5,
                    bgcolor: "rgba(5, 8, 13, 0.7)",
                  }}
                >
                  <CircularProgress />
                  <Typography sx={{ color: "#fff", mt: 1.5 }}>Estimating depth and building the point cloud...</Typography>
                </Box>
              )}
              <Canvas camera={{ position: [0, 0, 4], fov: 50 }} style={{ height: "100%", width: "100%" }}>
                <ScenePC
                  pcUrl={browserPcUrl}
                  objects={objects}
                  selectedObjectId={selectedObjectId}
                  intrinsics={projMeta?.intrinsics}
                  imageWidth={projMeta?.imageWidth}
                  imageHeight={projMeta?.imageHeight}
                  apiBase={API_BASE}
                  viewCommand={viewCommand}
                  onSelectedTranslationChange={updateSelectedTranslation}
                />
              </Canvas>
              {!browserPcUrl && !isLoading && (
                <Box
                  sx={{
                    position: "absolute",
                    inset: 0,
                    display: "grid",
                    placeItems: "center",
                    color: "#5f6e82",
                    pointerEvents: "none",
                  }}
                >
                  <Typography>The point cloud will appear here</Typography>
                </Box>
              )}
            </Box>
          </Paper>

          <Paper
            sx={{
              ...panelSx,
              p: 1.5,
              flex: "1 1 46%",
              minHeight: { xs: 360, lg: 0 },
              overflowY: "auto",
              position: "relative",
            }}
          >
            <Stack direction="row" spacing={0.75} alignItems="center" sx={{ mb: 1 }}>
              <ImageRoundedIcon fontSize="small" sx={{ color: "#8cc8ff" }} />
              <Typography sx={{ fontWeight: 800 }}>Preview and result</Typography>
            </Stack>
            {generateBusy && (
              <Box
                role="status"
                aria-live="polite"
                sx={{
                  position: "absolute",
                  inset: 0,
                  zIndex: 10,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  bgcolor: "rgba(5, 8, 13, 0.82)",
                  color: "#e8fff5",
                }}
              >
                <CircularProgress size={46} thickness={4} sx={{ color: "#58d6a2" }} />
                <Typography sx={{ mt: 1.5, fontWeight: 800 }}>Generating image...</Typography>
              </Box>
            )}
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: {
                  xs: "minmax(0, 1fr)",
                  sm: renderedUrl ? "repeat(2, minmax(0, 1fr))" : "minmax(0, 1fr)",
                },
                gap: 1.25,
                alignItems: "start",
              }}
            >
              {generatedUrl ? (
                <Box sx={{ minWidth: 0 }}>
                  <Typography variant="caption" sx={{ color: "#9ee8c5", fontWeight: 800 }}>
                    {lastGenerationSeed == null ? "PhyEdit result" : `PhyEdit result · seed ${lastGenerationSeed}`}
                  </Typography>
                  <Box
                    component="img"
                    src={generatedUrl}
                    alt="phyedit-result"
                    sx={{
                      display: "block",
                      width: "100%",
                      maxHeight: 280,
                      objectFit: "contain",
                      borderRadius: "8px",
                      border: "1px solid rgba(64, 211, 143, 0.32)",
                      bgcolor: "#05080d",
                      mt: 0.75,
                    }}
                  />
                </Box>
              ) : browserPcUrl ? (
                <Box sx={{ minWidth: 0 }}>{renderProjectionPreview}</Box>
              ) : (
                <Box
                  sx={{
                    p: 1.25,
                    border: "1px dashed rgba(148, 163, 184, 0.22)",
                    borderRadius: "8px",
                    color: "#74849a",
                    textAlign: "center",
                    fontSize: 13,
                  }}
                >
                  Upload an image to view the preview
                </Box>
              )}

              {renderedUrl && (
                <Box sx={{ minWidth: 0 }}>
                  <Typography variant="caption" sx={{ color: "#a8b4c5", fontWeight: 700 }}>
                    Geometric condition
                  </Typography>
                  <Box
                    component="img"
                    src={renderedUrl}
                    alt="moved"
                    sx={{
                      display: "block",
                      width: "100%",
                      maxHeight: 280,
                      objectFit: "contain",
                      borderRadius: "8px",
                      border: "1px solid rgba(148, 163, 184, 0.18)",
                      bgcolor: "#05080d",
                      mt: 0.75,
                    }}
                  />
                </Box>
              )}
            </Box>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
}

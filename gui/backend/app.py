import json
import os
import secrets
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import uvicorn
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

GUI_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = GUI_ROOT / ".env"

try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_FILE)
except ImportError:

    def _load_env_file(path: Path) -> None:
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and key not in os.environ:
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                os.environ[key] = val

    _load_env_file(_ENV_FILE)


def _env_str(key: str) -> Optional[str]:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return None
    v = str(v).strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v


try:
    from transformers import Sam3Model, Sam3Processor
except ImportError:  # pragma: no cover
    Sam3Model = None  # type: ignore
    Sam3Processor = None  # type: ignore

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from gui.backend.qwen_service import (  # noqa: E402
    QwenConfigurationError,
    QwenGenerationConfig,
    QwenGenerationService,
)
from phyedit.utils.geometry_utils import translate_masked_region_3d  # noqa: E402
from phyedit.utils.text_process import get_edit_prompt  # noqa: E402


def _resolve_project_path(value: Optional[str], default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_model_reference(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    local_path = PROJECT_ROOT / path
    return str(local_path.resolve()) if local_path.exists() else value


app = FastAPI(title="PhyEdit Geometry GUI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_ROOT = PROJECT_ROOT / "gui_runs"
TEMP_DIR = RUNS_ROOT / "temp"
SAVE_DIR = RUNS_ROOT / "save"
PREVIEW_PC_MAX_POINTS = int(os.getenv("GUI_PREVIEW_PC_MAX_POINTS", "90000"))
PREVIEW_PC_MAX_SIDE = int(os.getenv("GUI_PREVIEW_PC_MAX_SIDE", "480"))
PREVIEW_IMAGE_MAX_SIDE = int(os.getenv("GUI_PREVIEW_IMAGE_MAX_SIDE", "1024"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = TEMP_DIR
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

GUI_DEVICE = _env_str("GUI_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
PREVIEW_RENDER_DEVICE = _env_str("GUI_PREVIEW_RENDER_DEVICE") or GUI_DEVICE
DA3_MODEL_PATH = _env_str("DEPTH_MODEL_PATH") or _env_str("DEPTH_MODEL_ID") or "depth-anything/da3nested-giant-large"
SAM_MODEL_PATH = _env_str("SAM_MODEL_PATH") or "facebook/sam3"
QWEN_DEVICE = _env_str("QWEN_DEVICE") or GUI_DEVICE
QWEN_CONFIG_PATH = _resolve_project_path(_env_str("QWEN_CONFIG_PATH"), "configs/train_deepspeed.yaml")
QWEN_CHECKPOINT_PATH = (
    _resolve_project_path(_env_str("PHYEDIT_CHECKPOINT_PATH"), "") if _env_str("PHYEDIT_CHECKPOINT_PATH") else None
)
QWEN_MODEL_PATH = _resolve_model_reference(_env_str("QWEN_MODEL_PATH") or "Qwen/Qwen-Image-Edit-2511")
QWEN_BASE_AREA = int(_env_str("QWEN_BASE_AREA") or "589824")
QWEN_DEFAULT_STEPS = int(_env_str("QWEN_NUM_INFERENCE_STEPS") or "28")
QWEN_DEFAULT_GUIDANCE = float(_env_str("QWEN_GUIDANCE_SCALE") or "3.5")
GUI_HOST = _env_str("GUI_HOST") or "0.0.0.0"
GUI_PORT = int(_env_str("GUI_PORT") or "8000")

# session_id -> metadata (paths, shape)
SESSIONS: dict[str, dict[str, Any]] = {}

_da3_model = None
_sam_model = None
_sam_processor = None
_qwen_service = QwenGenerationService(
    QwenGenerationConfig(
        config_path=QWEN_CONFIG_PATH,
        checkpoint_path=QWEN_CHECKPOINT_PATH,
        pretrained_model_path=QWEN_MODEL_PATH,
        device=QWEN_DEVICE,
        base_area=QWEN_BASE_AREA,
    )
)


def get_da3():
    global _da3_model
    if _da3_model is None:
        _da3_model = DepthAnything3.from_pretrained(DA3_MODEL_PATH).to(GUI_DEVICE).eval()
    return _da3_model


def get_sam():
    global _sam_model, _sam_processor
    if Sam3Model is None or Sam3Processor is None:
        raise HTTPException(
            status_code=503,
            detail="SAM 3 is unavailable. Install a Transformers version with Sam3Model and Sam3Processor support.",
        )
    if _sam_model is None:
        _sam_model = Sam3Model.from_pretrained(SAM_MODEL_PATH).to(GUI_DEVICE)
        _sam_processor = Sam3Processor.from_pretrained(SAM_MODEL_PATH)
        _sam_model.eval()
    return _sam_model, _sam_processor, GUI_DEVICE


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": GUI_DEVICE,
        "preview": {
            "render_device": PREVIEW_RENDER_DEVICE,
            "max_points": PREVIEW_PC_MAX_POINTS,
            "max_side": PREVIEW_IMAGE_MAX_SIDE,
        },
        "da3_loaded": _da3_model is not None,
        "sam_loaded": _sam_model is not None,
        "qwen": _qwen_service.status(),
    }


def vis_point_to_camera_space(p_vis: np.ndarray) -> np.ndarray:
    """Convert view coordinates where p_vis = (x, -y_cam, -z_cam)."""
    return np.array([p_vis[0], -p_vis[1], -p_vis[2]], dtype=np.float64)


def project_vis_point_to_pixel(p_vis: np.ndarray, K: np.ndarray) -> Optional[tuple[float, float, float]]:
    """
    Convert frontend view coordinates back to the DA3 camera frame and project
    them with the same pinhole model used by ``geometry_utils``.

    u = fx * X / Z + cx, v = fy * Y / Z + cy
    """
    pc = vis_point_to_camera_space(np.asarray(p_vis, dtype=np.float64).reshape(3))
    z = float(pc[2])
    if z <= 1e-8:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * float(pc[0]) / z + cx
    v = fy * float(pc[1]) / z + cy
    return u, v, z


def ui_translation_to_delta_camera(dx_ui: float, dy_ui: float, dz_ui: float) -> tuple[float, float, float]:
    """Map a view-space translation to camera-space translation."""
    return (float(dx_ui), float(-dy_ui), float(-dz_ui))


def _has_nonempty_boxes(input_boxes: Optional[list]) -> bool:
    if not input_boxes or not isinstance(input_boxes, list):
        return False
    if len(input_boxes) < 1:
        return False
    inner = input_boxes[0]
    if not inner or not isinstance(inner, list):
        return False
    for b in inner:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return True
    return False


class SegmentRequest(BaseModel):
    session_id: str
    text: Optional[str] = None
    input_boxes: Optional[list] = None
    input_boxes_labels: Optional[list] = None


class RenderTranslateRequest(BaseModel):
    session_id: str
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rot_deg: float = 0.0


class ObjectEdit(BaseModel):
    object_id: str
    mask_url: str
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rot_deg: float = 0.0
    rot_v_deg: float = 0.0
    rot_h_deg: float = 0.0
    text: Optional[str] = None


class RenderMultiRequest(BaseModel):
    session_id: str
    objects: list[ObjectEdit]
    density: Optional[float] = None
    hole_fill_algo: bool = False
    hole_fill_rgb: Optional[list[int]] = None


class GenerateRequest(RenderMultiRequest):
    seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)
    num_inference_steps: int = Field(default=QWEN_DEFAULT_STEPS, ge=1, le=100)
    guidance_scale: float = Field(default=QWEN_DEFAULT_GUIDANCE, ge=0.0, le=20.0)
    additional_prompt: Optional[str] = None
    prompt_override: Optional[str] = None


class LoadSceneRequest(BaseModel):
    scene_dir: str
    recompute_depth: bool = True


class SaveSceneRequest(BaseModel):
    session_id: str
    objects: list[ObjectEdit]
    rendered_image_url: Optional[str] = None
    generated_image_url: Optional[str] = None
    note: Optional[str] = None


class ProjectRequest(BaseModel):
    session_id: str
    point_vis: list[float] = Field(..., min_length=3, max_length=3)


def _local_path_from_static_url(url: str) -> Path:
    filename = os.path.basename(url)
    return UPLOAD_DIR / filename


def _apply_object_edit(image_np_rgb: np.ndarray, meta: dict[str, Any], obj: ObjectEdit) -> np.ndarray:
    mp = _local_path_from_static_url(obj.mask_url)
    if not mp.is_file():
        raise HTTPException(status_code=400, detail=f"Mask does not exist: {obj.mask_url}")
    d_cam = (0.0, 0.0, float(obj.dz))
    shift_u_px = float(obj.dx) * float(max(meta["w"] - 1, 1))
    shift_v_px = float(obj.dy) * float(max(meta["h"] - 1, 1))
    result = translate_masked_region_3d(
        image=image_np_rgb,
        mask=str(mp),
        depth_map=meta["depth_path"],
        intrinsics=meta["K_path"],
        delta_camera_xyz=d_cam,
        shift_u_px=shift_u_px,
        shift_v_px=shift_v_px,
        rotation_deg=obj.rot_deg,
        lock_center_z=True,
        extrinsics=None,
        device=GUI_DEVICE,
    )
    final_images = result[0]
    return (final_images[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)


def _load_depth_fullres(meta: dict[str, Any]) -> np.ndarray:
    d = np.load(meta["depth_path"]).astype(np.float32)
    h, w = int(meta["h"]), int(meta["w"])
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d


def _load_point_cloud_for_session(
    session_id: str, *, preview: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc_path = UPLOAD_DIR / f"pc_preview_{session_id}.bin" if preview else UPLOAD_DIR / f"pc_{session_id}.bin"
    if preview and not pc_path.is_file():
        pc_path = UPLOAD_DIR / f"pc_{session_id}.bin"
    if not pc_path.is_file():
        raise HTTPException(
            status_code=400, detail="Point cloud is unavailable. Upload and reconstruct an image first."
        )
    raw = np.fromfile(str(pc_path), dtype=np.float32)
    if raw.size % 8 != 0:
        raise HTTPException(status_code=500, detail="Invalid point-cloud binary format")
    arr = raw.reshape(-1, 8)
    pos = arr[:, 0:3].astype(np.float32, copy=False)
    col = arr[:, 3:6].astype(np.float32, copy=False)
    uv = arr[:, 6:8].astype(np.float32, copy=False)
    return pos, col, uv


def _preview_point_indices(uv: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    n = int(uv.shape[0])
    if n <= PREVIEW_PC_MAX_POINTS:
        return np.arange(n, dtype=np.int64)

    max_side = max(int(image_w), int(image_h), 1)
    side_step = max(1, int(np.ceil(max_side / max(PREVIEW_PC_MAX_SIDE, 1))))
    count_step = max(1, int(np.ceil(np.sqrt(n / max(PREVIEW_PC_MAX_POINTS, 1)))))
    step = max(side_step, count_step)

    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    keep = (u % step == 0) & (v % step == 0)
    idx = np.flatnonzero(keep).astype(np.int64)
    if idx.size == 0:
        idx = np.linspace(0, n - 1, min(n, PREVIEW_PC_MAX_POINTS), dtype=np.int64)
    elif idx.size > PREVIEW_PC_MAX_POINTS:
        thin = np.linspace(0, idx.size - 1, PREVIEW_PC_MAX_POINTS, dtype=np.int64)
        idx = idx[thin]
    return idx


def _write_preview_point_cloud(session_id: str, combined: np.ndarray, image_w: int, image_h: int) -> tuple[str, int]:
    idx = _preview_point_indices(combined[:, 6:8], image_w, image_h)
    preview = combined[idx].astype(np.float32, copy=False)
    preview_path = UPLOAD_DIR / f"pc_preview_{session_id}.bin"
    preview.tofile(preview_path)
    return preview_path.name, int(preview.shape[0])


def _write_preview_image(session_id: str, image_bgr: np.ndarray) -> str:
    h, w = image_bgr.shape[:2]
    max_side = max(h, w, 1)
    scale = min(1.0, float(PREVIEW_IMAGE_MAX_SIDE) / float(max_side))
    if scale < 1.0:
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        preview = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        preview = image_bgr
    preview_name = f"preview_{session_id}.jpg"
    cv2.imwrite(str(UPLOAD_DIR / preview_name), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return preview_name


def _mask_flags_from_uv(uv: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    flags = np.zeros((uv.shape[0],), dtype=bool)
    idx = np.where(valid)[0]
    flags[idx] = mask_u8[v[idx], u[idx]] > 0
    return flags


def _mask_center_from_flags_uv(uv: np.ndarray, flags: np.ndarray) -> Optional[tuple[float, float]]:
    if flags.size == 0 or not np.any(flags):
        return None
    pts = uv[flags]
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _invert3x3(m: np.ndarray) -> Optional[np.ndarray]:
    det = float(np.linalg.det(m))
    if abs(det) < 1e-12:
        return None
    return np.linalg.inv(m).astype(np.float32)


def _rotate_uv_np(
    u: np.ndarray, v: np.ndarray, cx: float, cy: float, angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(angle_deg) or abs(angle_deg) < 1e-8:
        return u, v
    t = np.float32(angle_deg * np.pi / 180.0)
    ct = np.cos(t, dtype=np.float32)
    st = np.sin(t, dtype=np.float32)
    du = u - np.float32(cx)
    dv = v - np.float32(cy)
    return np.float32(cx) + du * ct - dv * st, np.float32(cy) + du * st + dv * ct


def _apply_object_transform_vis(
    positions: np.ndarray,
    flags: np.ndarray,
    center_uv: Optional[tuple[float, float]],
    dx: float,
    dy: float,
    dz: float,
    rot_deg: float,
    rot_v_deg: float,
    rot_h_deg: float,
    k9: np.ndarray,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    out = positions.copy()
    if not np.any(flags):
        return out

    inv_k = _invert3x3(k9)
    sel = np.where(flags)[0]
    z_cam_before = -out[sel, 2]
    valid_before = z_cam_before > 1e-6
    center_depth_before = float(np.mean(z_cam_before[valid_before])) if np.any(valid_before) else 1.0
    out[sel, 2] = out[sel, 2] - np.float32(dz)
    if inv_k is None or center_uv is None:
        return out

    z_cam = -out[sel, 2]
    valid_z = z_cam > 1e-6
    if not np.any(valid_z):
        return out

    cx0, cy0 = float(k9[0, 2]), float(k9[1, 2])

    x = out[sel, 0]
    y = out[sel, 1]
    z = out[sel, 2]
    X = x
    Y = -y
    Z = -z
    if center_uv is not None and (abs(float(rot_v_deg)) > 1e-8 or abs(float(rot_h_deg)) > 1e-8):
        fx = float(k9[0, 0])
        fy = float(k9[1, 1])
        cx = float(k9[0, 2])
        cy = float(k9[1, 2])
        zc = max(center_depth_before, 1e-6)
        Cx = ((float(center_uv[0]) - cx) * zc) / fx
        Cy = ((float(center_uv[1]) - cy) * zc) / fy
        Cz = zc
        Xc = X - np.float32(Cx)
        Yc = Y - np.float32(Cy)
        Zc = Z - np.float32(Cz)
        # Vertical axis in image plane -> camera Y axis rotation.
        if abs(float(rot_v_deg)) > 1e-8:
            tv = np.float32(float(rot_v_deg) * np.pi / 180.0)
            cv = np.cos(tv, dtype=np.float32)
            sv = np.sin(tv, dtype=np.float32)
            Xn = cv * Xc + sv * Zc
            Zn = -sv * Xc + cv * Zc
            Xc, Zc = Xn, Zn
        # Horizontal axis in image plane -> camera X axis rotation.
        if abs(float(rot_h_deg)) > 1e-8:
            th = np.float32(float(rot_h_deg) * np.pi / 180.0)
            ch = np.cos(th, dtype=np.float32)
            sh = np.sin(th, dtype=np.float32)
            Yn = ch * Yc - sh * Zc
            Zn = sh * Yc + ch * Zc
            Yc, Zc = Yn, Zn
        X = Xc + np.float32(Cx)
        Y = Yc + np.float32(Cy)
        Z = Zc + np.float32(Cz)
    proj_ok = Z > 1e-8
    if not np.any(proj_ok):
        return out

    u = (k9[0, 0] * X + k9[0, 1] * Y + k9[0, 2] * Z) / Z
    v = (k9[1, 0] * X + k9[1, 1] * Y + k9[1, 2] * Z) / Z

    if abs(float(dz)) > 1e-8:
        z_before = max(center_depth_before, 1e-6)
        z_after = max(z_before + float(dz), 1e-6)
        cu_after = ((float(center_uv[0]) - cx0) * z_before) / z_after + cx0
        cv_after = ((float(center_uv[1]) - cy0) * z_before) / z_after + cy0
        u = u + (np.float32(center_uv[0]) - np.float32(cu_after))
        v = v + (np.float32(center_uv[1]) - np.float32(cv_after))

    du_px = np.float32(dx * max(image_w - 1, 1))
    dv_px = np.float32(dy * max(image_h - 1, 1))
    u = u + du_px
    v = v + dv_px
    u, v = _rotate_uv_np(u, v, float(center_uv[0]) + float(du_px), float(center_uv[1]) + float(dv_px), float(rot_deg))

    nx = inv_k[0, 0] * u + inv_k[0, 1] * v + inv_k[0, 2]
    ny = inv_k[1, 0] * u + inv_k[1, 1] * v + inv_k[1, 2]
    nz = inv_k[2, 0] * u + inv_k[2, 1] * v + inv_k[2, 2]
    ok = np.abs(nz) > 1e-8
    if not np.any(ok):
        return out

    Xr = np.where(ok, (nx / nz) * Z, X)
    Yr = np.where(ok, (ny / nz) * Z, Y)
    Zr = Z
    out[sel, 0] = Xr.astype(np.float32)
    out[sel, 1] = (-Yr).astype(np.float32)
    out[sel, 2] = (-Zr).astype(np.float32)
    return out


def _fill_render_holes(out: np.ndarray, hit: np.ndarray, density: float) -> tuple[np.ndarray, np.ndarray]:
    if density < 3.0 or not np.any(hit):
        return out, hit
    k = int(np.clip(np.round(density / 3.0), 1, 4))
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    hit_dil = cv2.dilate(hit.astype(np.uint8), kernel, iterations=1).astype(bool)
    for ch in range(3):
        ch_img = out[:, :, ch]
        filled = cv2.blur(ch_img, (k * 2 + 1, k * 2 + 1))
        ch_img[hit_dil & (~hit)] = filled[hit_dil & (~hit)]
        out[:, :, ch] = ch_img
    return out, hit_dil


def _render_point_cloud_vis(
    positions: np.ndarray,
    colors: np.ndarray,
    k9: np.ndarray,
    w: int,
    h: int,
    density: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    hit = np.zeros((h, w), dtype=bool)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    X = positions[:, 0]
    Y = -positions[:, 1]
    Z = -positions[:, 2]
    valid_z = Z > 1e-7
    if not np.any(valid_z):
        return out, hit
    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]
    C = colors[valid_z]
    u = (k9[0, 0] * X + k9[0, 1] * Y + k9[0, 2] * Z) / Z
    v = (k9[1, 0] * X + k9[1, 1] * Y + k9[1, 2] * Z) / Z
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_view = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not np.any(in_view):
        return out, hit
    ui = ui[in_view]
    vi = vi[in_view]
    z = Z[in_view]
    c = C[in_view]
    rgb_all = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    order = np.argsort(z)
    ui = ui[order]
    vi = vi[order]
    z = z[order]
    rgb_all = rgb_all[order]

    # Depth-aware splat with user-controlled density scaling.
    density = float(np.clip(density, 0.5, 12.0))
    r_floor = int(np.clip(np.floor((density - 1.0) * 0.5), 0, 4))
    for px, py, pz, prgb in zip(ui, vi, z, rgb_all):
        if pz < 0.6:
            base_r = 2.0
        elif pz < 1.2:
            base_r = 1.0
        else:
            base_r = 0.6
        # Nonlinear boost at higher density values to make large settings effective.
        r = int(np.clip(np.round(base_r * (density**1.2)), 0, 14))
        r = max(r, r_floor)
        x0 = max(0, int(px) - r)
        x1 = min(w - 1, int(px) + r)
        y0 = max(0, int(py) - r)
        y1 = min(h - 1, int(py) + r)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if pz < depth[yy, xx]:
                    depth[yy, xx] = pz
                    out[yy, xx] = prgb
                    hit[yy, xx] = True
    return _fill_render_holes(out, hit, density)


def _render_point_cloud_vis_cuda(
    positions: np.ndarray,
    colors: np.ndarray,
    k9: np.ndarray,
    w: int,
    h: int,
    density: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    render_device = torch.device(device)
    if render_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA preview rendering is unavailable on {device}")

    density = float(np.clip(density, 0.5, 12.0))
    with torch.inference_mode():
        pos = torch.as_tensor(np.ascontiguousarray(positions), device=render_device, dtype=torch.float32)
        col = torch.as_tensor(np.ascontiguousarray(colors), device=render_device, dtype=torch.float32)
        k = torch.as_tensor(k9, device=render_device, dtype=torch.float32)

        x = pos[:, 0]
        y = -pos[:, 1]
        z = -pos[:, 2]
        valid = z > 1e-7
        x, y, z, col = x[valid], y[valid], z[valid], col[valid]
        if z.numel() == 0:
            return (
                np.full((h, w, 3), 255, dtype=np.uint8),
                np.zeros((h, w), dtype=bool),
            )

        u = (k[0, 0] * x + k[0, 1] * y + k[0, 2] * z) / z
        v = (k[1, 0] * x + k[1, 1] * y + k[1, 2] * z) / z
        ui = torch.round(u).to(torch.long)
        vi = torch.round(v).to(torch.long)
        in_view = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ui, vi, z, col = ui[in_view], vi[in_view], z[in_view], col[in_view]
        if z.numel() == 0:
            return (
                np.full((h, w, 3), 255, dtype=np.uint8),
                np.zeros((h, w), dtype=bool),
            )

        base_radius = torch.where(
            z < 0.6,
            torch.full_like(z, 2.0),
            torch.where(z < 1.2, torch.ones_like(z), torch.full_like(z, 0.6)),
        )
        radii = torch.round(base_radius * (density**1.2)).clamp(0, 14).to(torch.long)
        radius_floor = int(np.clip(np.floor((density - 1.0) * 0.5), 0, 4))
        radii = torch.maximum(radii, torch.tensor(radius_floor, device=render_device, dtype=torch.long))
        rgb = (col * 255.0).clamp(0, 255).to(torch.uint8)
        depth = torch.full((h * w,), float("inf"), device=render_device)
        out = torch.full((h * w, 3), 255, device=render_device, dtype=torch.uint8)
        hit = torch.zeros((h * w,), device=render_device, dtype=torch.bool)

        # Bound temporary candidate tensors even when the density slider is high.
        max_candidates = 2_000_000
        for radius in torch.unique(radii).detach().cpu().tolist():
            radius = int(radius)
            point_ids = torch.nonzero(radii == radius, as_tuple=False).flatten()
            side = radius * 2 + 1
            chunk_size = max(1, max_candidates // max(side * side, 1))
            offsets = torch.arange(-radius, radius + 1, device=render_device, dtype=torch.long)
            offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
            offset_x = offset_x.flatten()
            offset_y = offset_y.flatten()

            for start in range(0, point_ids.numel(), chunk_size):
                ids = point_ids[start : start + chunk_size]
                xx = ui[ids, None] + offset_x[None, :]
                yy = vi[ids, None] + offset_y[None, :]
                inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                expanded_ids = ids[:, None].expand_as(xx)
                pixel_ids = (yy * w + xx)[inside]
                candidate_ids = expanded_ids[inside]
                if pixel_ids.numel() == 0:
                    continue
                candidate_depth = z[candidate_ids]
                depth.scatter_reduce_(0, pixel_ids, candidate_depth, reduce="amin", include_self=True)
                winners = candidate_depth <= depth[pixel_ids] + 1e-7
                winner_pixels = pixel_ids[winners]
                winner_points = candidate_ids[winners]
                out[winner_pixels] = rgb[winner_points]
                hit[winner_pixels] = True

        out_np = out.reshape(h, w, 3).cpu().numpy()
        hit_np = hit.reshape(h, w).cpu().numpy()
    return _fill_render_holes(out_np, hit_np, density)


def _render_preview_point_cloud(
    positions: np.ndarray,
    colors: np.ndarray,
    k9: np.ndarray,
    w: int,
    h: int,
    density: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        if torch.device(PREVIEW_RENDER_DEVICE).type == "cuda":
            out, hit = _render_point_cloud_vis_cuda(positions, colors, k9, w, h, density, PREVIEW_RENDER_DEVICE)
            return out, hit, "cuda"
    except (RuntimeError, ValueError) as exc:
        print(f"[GUI][preview] CUDA renderer unavailable, using CPU: {exc}")
    if len(positions) > PREVIEW_PC_MAX_POINTS:
        indices = np.linspace(0, len(positions) - 1, PREVIEW_PC_MAX_POINTS, dtype=np.int64)
        positions = positions[indices]
        colors = colors[indices]
    out, hit = _render_point_cloud_vis(positions, colors, k9, w, h, density)
    return out, hit, "cpu"


def _estimate_auto_density(positions: np.ndarray, k9: np.ndarray, w: int, h: int, edited_flags: np.ndarray) -> float:
    if edited_flags.size == 0 or not np.any(edited_flags):
        return 2.5
    pts = positions[edited_flags]
    X = pts[:, 0]
    Y = -pts[:, 1]
    Z = -pts[:, 2]
    valid = Z > 1e-7
    if not np.any(valid):
        return 2.5
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    u = (k9[0, 0] * X + k9[0, 1] * Y + k9[0, 2] * Z) / Z
    v = (k9[1, 0] * X + k9[1, 1] * Y + k9[1, 2] * Z) / Z
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_view = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not np.any(in_view):
        return 2.5
    ui = ui[in_view]
    vi = vi[in_view]
    xmin, xmax = int(ui.min()), int(ui.max())
    ymin, ymax = int(vi.min()), int(vi.max())
    area = max((xmax - xmin + 1) * (ymax - ymin + 1), 1)
    uniq = np.unique(vi * w + ui).size
    coverage = float(uniq) / float(area)
    # lower projected coverage => higher density
    if coverage >= 0.22:
        return 2.0
    d = 2.0 + (0.22 - coverage) * 55.0
    return float(np.clip(d, 2.0, 12.0))


def _transform_positions_by_objects(
    session_id: str,
    meta: dict[str, Any],
    objects: list[ObjectEdit],
    *,
    preview: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = int(meta["w"]), int(meta["h"])
    k9 = np.load(meta["K_path"]).astype(np.float32).reshape(3, 3)
    positions, colors, uv = _load_point_cloud_for_session(session_id, preview=preview)
    transformed = positions.copy()
    for obj in objects:
        mp = _local_path_from_static_url(obj.mask_url)
        if not mp.is_file():
            raise HTTPException(status_code=400, detail=f"Mask does not exist: {obj.mask_url}")
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=400, detail=f"Cannot read mask: {obj.mask_url}")
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        flags = _mask_flags_from_uv(uv, m)
        center = _mask_center_from_flags_uv(uv, flags)
        transformed = _apply_object_transform_vis(
            transformed,
            flags,
            center,
            float(obj.dx),
            float(obj.dy),
            float(obj.dz),
            float(obj.rot_deg),
            float(obj.rot_v_deg),
            float(obj.rot_h_deg),
            k9,
            w,
            h,
        )
    return transformed, colors


def _object_center_state(meta: dict[str, Any], obj: ObjectEdit) -> tuple[list[float], list[float]]:
    mask_path = _local_path_from_static_url(obj.mask_url)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return [0.5, 0.5, 1.0], [0.5, 0.5, 1.0]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return [0.5, 0.5, 1.0], [0.5, 0.5, 1.0]

    d = np.load(meta["depth_path"]).astype(np.float32)
    z_vals = d[ys, xs].astype(np.float64)
    z_vals = z_vals[z_vals > 1e-8]
    z_med = float(np.median(z_vals)) if len(z_vals) > 0 else 1.0
    u0 = float(np.median(xs.astype(np.float64)))
    v0 = float(np.median(ys.astype(np.float64)))

    shift_u_px = float(obj.dx) * float(max(meta["w"] - 1, 1))
    shift_v_px = float(obj.dy) * float(max(meta["h"] - 1, 1))
    u1 = u0 + shift_u_px
    v1 = v0 + shift_v_px
    z1 = max(z_med + float(obj.dz), 1e-8)

    before = [
        float(np.clip(u0 / max(meta["w"] - 1, 1), 0.0, 1.0)),
        float(np.clip(v0 / max(meta["h"] - 1, 1), 0.0, 1.0)),
        float(z_med),
    ]
    after = [
        float(np.clip(u1 / max(meta["w"] - 1, 1), 0.0, 1.0)),
        float(np.clip(v1 / max(meta["h"] - 1, 1), 0.0, 1.0)),
        float(z1),
    ]
    return before, after


def _create_session_from_bgr_image(img_bgr: np.ndarray, session_id: str) -> dict[str, Any]:
    if img_bgr is None or img_bgr.ndim != 3:
        raise HTTPException(status_code=400, detail="Invalid image data")
    h, w = img_bgr.shape[:2]

    color_filename = f"orig_{session_id}.png"
    color_path = UPLOAD_DIR / color_filename
    cv2.imwrite(str(color_path), img_bgr)
    preview_image_name = _write_preview_image(session_id, img_bgr)
    process_res = max(h, w)

    pred_info = get_da3().inference([str(color_path)], process_res=process_res)
    depth_np = np.asarray(pred_info.depth)
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    Hm, Wm = int(depth_np.shape[-2]), int(depth_np.shape[-1])
    scale_w, scale_h = w / Wm, h / Hm

    dt = torch.from_numpy(np.asarray(pred_info.depth)).float()
    if dt.ndim == 2:
        dt = dt.unsqueeze(0)
    dt = dt[:1].unsqueeze(1)
    depth_upsampled = torch.nn.functional.interpolate(dt, size=(h, w), mode="bilinear", align_corners=True)
    depth_upsampled = depth_upsampled.unsqueeze(-1)

    K_raw = np.asarray(pred_info.intrinsics)
    if K_raw.ndim == 3:
        K_raw = K_raw[0]
    K_scaled = K_raw.astype(np.float32).copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 2] *= scale_h

    K_t = torch.from_numpy(K_scaled).float()
    if K_t.ndim == 2:
        ixt_tensor = K_t.unsqueeze(0).unsqueeze(0)
    elif K_t.ndim == 3:
        ixt_tensor = K_t[:1].unsqueeze(0)
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected intrinsics shape: {tuple(K_t.shape)}")

    world_points = unproject_depth(depth_upsampled, ixt_tensor, None)
    points_3d = world_points.squeeze().reshape(-1, 3).numpy().astype(np.float32)
    points_3d[:, 1] = -points_3d[:, 1]
    points_3d[:, 2] = -points_3d[:, 2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colors = (img_rgb.reshape(-1, 3) / 255.0).astype(np.float32)
    idx = np.arange(h * w, dtype=np.float32)
    u_coord = (idx % w).astype(np.float32)
    v_coord = (idx // w).astype(np.float32)
    uv = np.stack([u_coord, v_coord], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    ind_arr = np.asarray(list(ind), dtype=np.int64)
    uv_sel = uv[ind_arr]

    combined = np.hstack((pts, cols, uv_sel)).astype(np.float32)
    pc_path = UPLOAD_DIR / f"pc_{session_id}.bin"
    combined.tofile(pc_path)
    pc_preview_name, pc_preview_count = _write_preview_point_cloud(session_id, combined, w, h)

    depth_hw = depth_upsampled.squeeze().cpu().numpy().astype(np.float32)
    depth_valid = depth_hw[np.isfinite(depth_hw)]
    if depth_valid.size > 0:
        depth_min = float(np.min(depth_valid))
        depth_max = float(np.max(depth_valid))
        depth_p05 = float(np.percentile(depth_valid, 5))
        depth_p95 = float(np.percentile(depth_valid, 95))
    else:
        depth_min = 0.0
        depth_max = 1.0
        depth_p05 = 0.0
        depth_p95 = 1.0
    depth_path = UPLOAD_DIR / f"depth_{session_id}.npy"
    np.save(str(depth_path), depth_hw)

    K_path = UPLOAD_DIR / f"K_{session_id}.npy"
    np.save(str(K_path), K_scaled)

    SESSIONS[session_id] = {
        "h": h,
        "w": w,
        "color_path": str(color_path),
        "depth_path": str(depth_path),
        "depth_range": [depth_min, depth_max],
        "K_path": str(K_path),
        "mask_path": None,
    }

    return {
        "session_id": session_id,
        "pc_url": f"/static/pc_{session_id}.bin",
        "pc_preview_url": f"/static/{pc_preview_name}",
        "pc_point_count": int(combined.shape[0]),
        "pc_preview_point_count": pc_preview_count,
        "image_key": f"/static/orig_{session_id}.png",
        "preview_image_url": f"/static/{preview_image_name}",
        "intrinsics": K_scaled.astype(float).reshape(-1).tolist(),
        "image_width": w,
        "image_height": h,
        "depth_stats": {
            "min": depth_min,
            "max": depth_max,
            "p05": depth_p05,
            "p95": depth_p95,
        },
    }


@app.post("/predict-depth")
async def predict_depth(image: UploadFile = File(...)):
    session_id = uuid.uuid4().hex[:8]
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return _create_session_from_bgr_image(img, session_id=session_id)


@app.post("/load-scene")
async def load_scene(req: LoadSceneRequest):
    # only allow loading from gui_runs/save/*
    scene_root = (SAVE_DIR / req.scene_dir).resolve()
    if SAVE_DIR.resolve() not in scene_root.parents and scene_root != SAVE_DIR.resolve():
        raise HTTPException(status_code=400, detail="scene_dir must be inside gui_runs/save")
    scene_json = scene_root / "scene.json"
    if not scene_json.is_file():
        raise HTTPException(status_code=404, detail=f"scene.json does not exist: {scene_json}")
    try:
        scene = json.loads(scene_json.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot parse scene.json")
    orig_path = scene_root / "orig.png"
    if not orig_path.is_file():
        raise HTTPException(status_code=404, detail=f"orig.png does not exist: {orig_path}")

    img_bgr = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Cannot read orig.png")

    session_id = uuid.uuid4().hex[:8]
    if req.recompute_depth:
        base = _create_session_from_bgr_image(img_bgr, session_id=session_id)
    else:
        base = _create_session_from_bgr_image(img_bgr, session_id=session_id)
    # restore objects: copy masks into temp/static and return params
    objects_out: list[dict[str, Any]] = []
    for obj in scene.get("objects", []):
        mask_file = obj.get("mask_file")
        if not mask_file:
            continue
        mask_src = (scene_root / str(mask_file)).resolve()
        if not mask_src.is_file():
            continue
        mask_name = f"seg_{session_id}_{uuid.uuid4().hex[:6]}.png"
        mask_dst = UPLOAD_DIR / mask_name
        shutil.copy2(mask_src, mask_dst)

        params = obj.get("params") or {}
        objects_out.append(
            {
                "object_id": obj.get("object_id") or uuid.uuid4().hex[:8],
                "mask_url": f"/static/{mask_name}",
                "text": obj.get("text") or "",
                "translation": {
                    "x": float(params.get("dx", 0.0)),
                    "y": float(params.get("dy", 0.0)),
                    "z": float(params.get("dz", 0.0)),
                },
                "rotation_deg": float(params.get("rot_deg", 0.0)),
                "rotation_v_deg": float(params.get("rot_v_deg", 0.0)),
                "rotation_h_deg": float(params.get("rot_h_deg", 0.0)),
            }
        )

    base["objects"] = objects_out
    return base


@app.post("/segment")
async def segment(req: SegmentRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[req.session_id]
    image = Image.open(meta["color_path"]).convert("RGB")

    has_text = bool(req.text and str(req.text).strip())
    has_boxes = _has_nonempty_boxes(req.input_boxes)

    if not has_text and not has_boxes:
        raise HTTPException(status_code=400, detail="Provide a text prompt, a bounding box, or both")

    model, processor, device = get_sam()

    kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
    if has_text:
        kwargs["text"] = str(req.text).strip()
    if has_boxes:
        ib = req.input_boxes
        assert ib is not None
        kwargs["input_boxes"] = ib
        if req.input_boxes_labels is not None:
            kwargs["input_boxes_labels"] = req.input_boxes_labels
        else:
            kwargs["input_boxes_labels"] = [[1] * len(ib[0])]

    inputs = processor(**kwargs)
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if target_sizes is None:
        raise HTTPException(status_code=500, detail="SAM processor did not return original_sizes")
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=target_sizes.tolist(),
    )[0]

    masks = results.get("masks", [])
    if len(masks) == 0:
        meta["mask_path"] = None
        return {"mask_url": None, "num_instances": 0, "message": "No instance detected"}

    h, w = meta["h"], meta["w"]
    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        arr = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        arr = (arr > 0.5).astype(np.uint8)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, arr * 255)

    mask_name = f"seg_{req.session_id}_{uuid.uuid4().hex[:6]}.png"
    mask_path = UPLOAD_DIR / mask_name
    cv2.imwrite(str(mask_path), union)
    meta["mask_path"] = str(mask_path)

    return {
        "mask_url": f"/static/{mask_name}",
        "num_instances": len(masks),
    }


@app.post("/upload-mask")
async def upload_mask(session_id: str = Form(...), mask: UploadFile = File(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[session_id]
    raw = await mask.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Mask file is empty")

    arr = np.frombuffer(raw, dtype=np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise HTTPException(status_code=400, detail="Cannot decode mask image")

    if m.ndim == 3:
        if m.shape[2] == 4:
            m = m[:, :, 3]
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    elif m.ndim != 2:
        raise HTTPException(status_code=400, detail="Unexpected mask dimensions")

    h, w = int(meta["h"]), int(meta["w"])
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m_bin = (m > 127).astype(np.uint8) * 255
    if int(np.count_nonzero(m_bin)) == 0:
        raise HTTPException(status_code=400, detail="Mask contains no foreground pixels")

    mask_name = f"seg_{session_id}_{uuid.uuid4().hex[:6]}.png"
    mask_path = UPLOAD_DIR / mask_name
    cv2.imwrite(str(mask_path), m_bin)
    return {"mask_url": f"/static/{mask_name}", "num_pixels": int(np.count_nonzero(m_bin))}


@app.post("/upload-masks")
async def upload_masks(session_id: str = Form(...), masks: list[UploadFile] = File(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[session_id]
    if not masks or len(masks) == 0:
        raise HTTPException(status_code=400, detail="Upload at least one mask")

    h, w = int(meta["h"]), int(meta["w"])

    out_urls: list[str] = []
    out_pixels: list[int] = []
    for mi, mask in enumerate(masks):
        raw = await mask.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"mask[{mi}] is empty")
        arr = np.frombuffer(raw, dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise HTTPException(status_code=400, detail=f"Cannot decode mask[{mi}]")

        if m.ndim == 3:
            if m.shape[2] == 4:
                m = m[:, :, 3]
            else:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        elif m.ndim != 2:
            raise HTTPException(status_code=400, detail=f"Unexpected dimensions for mask[{mi}]")

        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        m_bin = (m > 127).astype(np.uint8) * 255
        nz = int(np.count_nonzero(m_bin))
        if nz == 0:
            raise HTTPException(status_code=400, detail=f"mask[{mi}] contains no foreground pixels")

        mask_name = f"seg_{session_id}_{uuid.uuid4().hex[:6]}_{mi:02d}.png"
        mask_path = UPLOAD_DIR / mask_name
        cv2.imwrite(str(mask_path), m_bin)

        out_urls.append(f"/static/{mask_name}")
        out_pixels.append(nz)

    return {"mask_urls": out_urls, "num_pixels": out_pixels}


@app.post("/clear-segment")
async def clear_segment(req: SegmentRequest):
    """Remove server-side segmentation masks and session references."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[req.session_id]
    mp = meta.get("mask_path")
    if mp and os.path.isfile(mp):
        try:
            os.remove(mp)
        except OSError:
            pass
    meta["mask_path"] = None
    return {"ok": True}


@app.post("/render-translate")
async def render_translate(req: RenderTranslateRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[req.session_id]
    mp = meta.get("mask_path")
    if not mp or not os.path.isfile(mp):
        raise HTTPException(status_code=400, detail="Run segmentation before rendering")

    d_cam = (0.0, 0.0, float(req.dz))
    shift_u_px = float(req.dx) * float(max(meta["w"] - 1, 1))
    shift_v_px = float(req.dy) * float(max(meta["h"] - 1, 1))
    out_id = uuid.uuid4().hex[:8]
    out_path = UPLOAD_DIR / f"moved_{req.session_id}_{out_id}.png"

    result = translate_masked_region_3d(
        image=meta["color_path"],
        mask=mp,
        depth_map=meta["depth_path"],
        intrinsics=meta["K_path"],
        delta_camera_xyz=d_cam,
        shift_u_px=shift_u_px,
        shift_v_px=shift_v_px,
        rotation_deg=req.rot_deg,
        lock_center_z=True,
        extrinsics=None,
        device=GUI_DEVICE,
    )
    final_images = result[0]

    arr = (final_images[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(out_path))

    return {"image_url": f"/static/{out_path.name}"}


def _render_moved_image(req: RenderMultiRequest) -> Path:
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    if len(req.objects) == 0:
        raise HTTPException(status_code=400, detail="objects cannot be empty")
    meta = SESSIONS[req.session_id]
    w, h = int(meta["w"]), int(meta["h"])
    orig_img = np.array(Image.open(meta["color_path"]).convert("RGB"))
    k9 = np.load(meta["K_path"]).astype(np.float32).reshape(3, 3)
    positions, colors, uv = _load_point_cloud_for_session(req.session_id)
    transformed = positions.copy()
    edited_union_flags = np.zeros((positions.shape[0],), dtype=bool)
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for obj in req.objects:
        mp = _local_path_from_static_url(obj.mask_url)
        if not mp.is_file():
            raise HTTPException(status_code=400, detail=f"Mask does not exist: {obj.mask_url}")
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=400, detail=f"Cannot read mask: {obj.mask_url}")
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        union_mask = np.maximum(union_mask, (m > 0).astype(np.uint8) * 255)
        flags = _mask_flags_from_uv(uv, m)
        edited_union_flags |= flags
        center = _mask_center_from_flags_uv(uv, flags)
        transformed = _apply_object_transform_vis(
            transformed,
            flags,
            center,
            float(obj.dx),
            float(obj.dy),
            float(obj.dz),
            float(obj.rot_deg),
            float(obj.rot_v_deg),
            float(obj.rot_h_deg),
            k9,
            w,
            h,
        )

    density = (
        float(req.density)
        if req.density is not None
        else _estimate_auto_density(transformed, k9, w, h, edited_union_flags)
    )
    proj_img, proj_hit = _render_point_cloud_vis(transformed, colors, k9, w, h, density)
    base_img = orig_img.copy()
    fill_rgb = req.hole_fill_rgb if req.hole_fill_rgb is not None else [255, 255, 255]
    if len(fill_rgb) < 3:
        fill_rgb = [255, 255, 255]
    fill_rgb = [int(np.clip(int(v), 0, 255)) for v in fill_rgb[:3]]
    if req.hole_fill_algo:
        # Fill the "moved-away" regions using local inpainting.
        base_img = cv2.inpaint(base_img, union_mask, 3, cv2.INPAINT_TELEA)
    else:
        # Keep old behavior: moved-away regions are rendered as a solid color.
        base_img[union_mask > 0] = np.array(fill_rgb, dtype=np.uint8)
    img = base_img
    img[proj_hit] = proj_img[proj_hit]
    out_id = uuid.uuid4().hex[:8]
    out_path = UPLOAD_DIR / f"moved_multi_{req.session_id}_{out_id}.png"
    Image.fromarray(img).save(str(out_path))
    return out_path


@app.post("/render-multi")
async def render_multi(req: RenderMultiRequest):
    out_path = _render_moved_image(req)
    return {"image_url": f"/static/{out_path.name}"}


def _generation_coordinates(
    meta: dict[str, Any], objects: list[ObjectEdit]
) -> tuple[list[list[float]], list[list[float]], list[list[list[float]]], list[float]]:
    depth_range = meta.get("depth_range")
    if not isinstance(depth_range, list) or len(depth_range) != 2:
        depth = np.load(meta["depth_path"]).astype(np.float32)
        valid = depth[np.isfinite(depth)]
        depth_range = [float(valid.min()), float(valid.max())] if valid.size else [0.0, 1.0]

    depth_min, depth_max = float(depth_range[0]), float(depth_range[1])
    if depth_max - depth_min <= 1e-8:
        depth_max = depth_min + 1.0

    source_coordinates: list[list[float]] = []
    target_coordinates: list[list[float]] = []
    prompt_coordinates: list[list[list[float]]] = []
    for obj in objects:
        source, target = _object_center_state(meta, obj)
        source_coordinates.append(source)
        target_coordinates.append(target)

        prompt_source = source.copy()
        prompt_target = target.copy()
        prompt_source[2] = (prompt_source[2] - depth_min) / (depth_max - depth_min)
        prompt_target[2] = (prompt_target[2] - depth_min) / (depth_max - depth_min)
        prompt_coordinates.append([prompt_source, prompt_target])

    return source_coordinates, target_coordinates, prompt_coordinates, [depth_min, depth_max]


@app.post("/generate")
def generate_final(req: GenerateRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    if not req.objects:
        raise HTTPException(status_code=400, detail="objects cannot be empty")

    seed = req.seed if req.seed is not None else secrets.randbelow(2_147_483_648)
    meta = SESSIONS[req.session_id]
    mask_paths = []
    object_names = []
    for index, obj in enumerate(req.objects):
        mask_path = _local_path_from_static_url(obj.mask_url)
        if not mask_path.is_file():
            raise HTTPException(status_code=400, detail=f"Mask does not exist: {obj.mask_url}")
        mask_paths.append(str(mask_path))
        object_names.append((obj.text or "").strip() or f"object {index + 1}")

    source_coordinates, target_coordinates, prompt_coordinates, depth_range = _generation_coordinates(meta, req.objects)
    prompt = (req.prompt_override or "").strip()
    if not prompt:
        prompt = get_edit_prompt(
            object_name=object_names,
            coordinates=prompt_coordinates,
            additional_prompt=req.additional_prompt,
            xy_coord_range="zero_1",
        )

    moved_path = _render_moved_image(req)
    try:
        generated_images = _qwen_service.generate(
            src_image=[meta["color_path"]],
            mask_image=[mask_paths],
            depth_image=[meta["depth_path"]],
            intrinsics=[meta["K_path"]],
            extrinsics=[np.eye(4, dtype=np.float32)],
            src_obj_coords=[source_coordinates],
            target_obj_coords=[target_coordinates],
            image_depth_range=[depth_range],
            objects=[object_names],
            prompt_override=[prompt],
            need_moved_image=True,
            moved_image_input=[str(moved_path)],
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=seed,
        )
    except QwenConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail=(
                f"Qwen generation ran out of memory on {QWEN_DEVICE}. "
                "Use a different QWEN_DEVICE or reduce competing GPU memory usage."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PhyEdit generation failed: {exc}") from exc

    if not generated_images:
        raise HTTPException(status_code=500, detail="PhyEdit generation returned no images")
    output = generated_images[0]
    if not isinstance(output, Image.Image):
        raise HTTPException(status_code=500, detail="PhyEdit generation returned an invalid image")

    output_name = f"phyedit_{req.session_id}_{uuid.uuid4().hex[:8]}_seed{seed}.png"
    output_path = UPLOAD_DIR / output_name
    output.convert("RGB").save(output_path)
    return {
        "image_url": f"/static/{output_name}",
        "moved_image_url": f"/static/{moved_path.name}",
        "prompt": prompt,
        "seed": seed,
        "random_seed": req.seed is None,
        "device": QWEN_DEVICE,
    }


@app.post("/preview-render")
def preview_render(req: RenderMultiRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    if len(req.objects) == 0:
        raise HTTPException(status_code=400, detail="objects cannot be empty")
    meta = SESSIONS[req.session_id]
    w, h = int(meta["w"]), int(meta["h"])
    k9 = np.load(meta["K_path"]).astype(np.float32).reshape(3, 3)
    started = time.perf_counter()
    transformed, colors = _transform_positions_by_objects(req.session_id, meta, req.objects)
    density = float(req.density) if req.density is not None else 2.5
    max_side = max(w, h, 1)
    scale = min(1.0, float(PREVIEW_IMAGE_MAX_SIDE) / float(max_side))
    preview_w = max(1, int(round(w * scale)))
    preview_h = max(1, int(round(h * scale)))
    preview_k = k9.copy()
    preview_k[0, :] *= np.float32(preview_w / max(w, 1))
    preview_k[1, :] *= np.float32(preview_h / max(h, 1))

    img_rgb, _, renderer = _render_preview_point_cloud(transformed, colors, preview_k, preview_w, preview_h, density)
    preview_name = _write_preview_image(req.session_id, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"[GUI][preview] session={req.session_id} renderer={renderer} "
        f"points={len(transformed)} size={preview_w}x{preview_h} "
        f"elapsed={elapsed_ms:.1f}ms"
    )
    return {
        "image_url": f"/static/{preview_name}",
        "renderer": renderer,
        "point_count": len(transformed),
        "render_ms": round(elapsed_ms, 1),
    }


@app.post("/save-scene")
async def save_scene(req: SaveSceneRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[req.session_id]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SAVE_DIR / f"{ts}_{req.session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dst = out_dir / "orig.png"
    shutil.copy2(meta["color_path"], src_dst)

    rendered_url = req.rendered_image_url
    if rendered_url:
        rendered_path = _local_path_from_static_url(rendered_url)
        if rendered_path.is_file():
            shutil.copy2(rendered_path, out_dir / "moved_render.png")

    generated_url = req.generated_image_url
    if generated_url:
        generated_path = _local_path_from_static_url(generated_url)
        if generated_path.is_file():
            shutil.copy2(generated_path, out_dir / "phyedit_result.png")

    objects_manifest = []
    for i, obj in enumerate(req.objects):
        mask_src = _local_path_from_static_url(obj.mask_url)
        mask_name = f"mask_{i:02d}_{obj.object_id}.png"
        if mask_src.is_file():
            shutil.copy2(mask_src, out_dir / mask_name)
        center_before, center_after = _object_center_state(meta, obj)
        objects_manifest.append(
            {
                "object_id": obj.object_id,
                "text": obj.text,
                "mask_file": mask_name,
                "params": {
                    "dx": obj.dx,
                    "dy": obj.dy,
                    "dz": obj.dz,
                    "rot_deg": obj.rot_deg,
                    "rot_v_deg": obj.rot_v_deg,
                    "rot_h_deg": obj.rot_h_deg,
                },
                "center_before": center_before,
                "center_after": center_after,
            }
        )

    # Save moved point cloud (.ply) for downstream use/debug.
    moved_ply_name = "moved_points.ply"
    moved_ply_path = out_dir / moved_ply_name
    try:
        moved_pos, moved_col = _transform_positions_by_objects(req.session_id, meta, req.objects)
        moved_pcd = o3d.geometry.PointCloud()
        moved_pcd.points = o3d.utility.Vector3dVector(moved_pos.astype(np.float64))
        moved_pcd.colors = o3d.utility.Vector3dVector(moved_col.astype(np.float64))
        o3d.io.write_point_cloud(str(moved_ply_path), moved_pcd, write_ascii=False, compressed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save moved point cloud: {e}")

    manifest = {
        "session_id": req.session_id,
        "saved_at": ts,
        "note": req.note,
        "image_width": meta["w"],
        "image_height": meta["h"],
        "coord_format": "x_norm_y_norm_z_depth",
        "depth_path": meta["depth_path"],
        "intrinsics_path": meta["K_path"],
        "moved_pointcloud_file": moved_ply_name,
        "objects": objects_manifest,
    }
    with open(out_dir / "scene.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {"save_dir": str(out_dir), "scene_json": str(out_dir / "scene.json")}


@app.post("/project-point")
async def project_point(req: ProjectRequest):
    """Project a view-space 3D point into the image for geometry validation."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    meta = SESSIONS[req.session_id]
    K = np.load(meta["K_path"])
    out = project_vis_point_to_pixel(np.array(req.point_vis, dtype=np.float64), K)
    if out is None:
        raise HTTPException(status_code=400, detail="Point is invalid or behind the camera")
    u, v, z = out
    return {"u": u, "v": v, "z": z, "inside": bool(0 <= u < meta["w"] and 0 <= v < meta["h"])}


if __name__ == "__main__":
    uvicorn.run(app, host=GUI_HOST, port=GUI_PORT)

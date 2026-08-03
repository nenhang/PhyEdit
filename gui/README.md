# 🎛️ PhyEdit Geometry GUI

The GUI provides an interactive PhyEdit workflow. It reconstructs a point cloud
with Depth Anything 3, segments objects with SAM 3, supports 3D translation and
rotation, renders the geometric condition, and generates the final edited image
with the public PhyEdit checkpoint.

## 📦 Install

From the repository root:

The Vite frontend requires Node.js `20.19+` or `22.12+`.

```bash
pip install -r gui/requirements.txt
cp gui/.env.example gui/.env

hf download ruihangxu/PhyEdit phyedit_lora.safetensors \
  --local-dir checkpoints/PhyEdit

cd gui/frontend
npm ci
cp .env.example .env
```

Set model paths and devices in `gui/.env`. The backend loads DA3 on the first
image upload, SAM 3 on the first segmentation request, and Qwen Image Edit on
the first final-generation request.

## ▶️ Run

Start the backend from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
GUI_DEVICE=cuda:0 \
QWEN_DEVICE=cuda:0 \
bash gui/backend/run_server.sh
```

This places DA3, SAM 3, and Qwen Image Edit on physical GPU 0. To place the
geometry models on physical GPU 0 and Qwen on physical GPU 1, use:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
GUI_DEVICE=cuda:0 \
QWEN_DEVICE=cuda:1 \
bash gui/backend/run_server.sh
```

CUDA indices are logical within `CUDA_VISIBLE_DEVICES`. Keeping all three
models on one GPU requires enough free VRAM; use separate devices otherwise.

In a second terminal, start the frontend:

```bash
cd gui/frontend
npm run dev
```

Open `http://localhost:5173`. Generated sessions and saved scenes are written
to `gui_runs/`, which is ignored by Git. The health endpoint at
`http://localhost:8000/health` reports the configured devices and lazy-loading
state for all three models. During development, Vite proxies browser requests
from `/api` to the backend on port `8000`, so only the frontend address needs
to be exposed to the browser.

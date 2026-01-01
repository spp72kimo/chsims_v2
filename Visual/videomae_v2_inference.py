# ==========================================
# VideoMAE V2 Valence Inference Script (Multi-Clip)
# 用於 CH-SIMS-V2 資料集的 Valence 預測
# 特色:
# - Multi-clip inference (mean pooling over clips)
# - ASD bbox cache crop (active_speaker / fallback_face)
# - 影片解碼 + 後處理(image_processor) timeout 防護
# - timeout / error 自動寫回 bad_videos.json (下次直接跳過)
# - MAX_TIMEOUTS / MAX_OTHER_ERRORS 保護，避免越跑越慢或卡死
# ==========================================

# ==========================================
# Step 0. 安裝套件（Colab 需要先跑這段）
# ==========================================
# !pip install -q "transformers[torch]" decord

# ==========================================
# Step 1. 掛載 Google Drive 並準備資料
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import zipfile
from tqdm import tqdm

def copy_with_progress(src, dst, buffer_size=1024*1024*16):
    total_size = os.path.getsize(src)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Copying zip") as pbar:
            while True:
                buf = fsrc.read(buffer_size)
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

# ===== 設定路徑 =====
target_dir = '/content/datasets'
extract_dir = os.path.join(target_dir, 'CH-SIMS-V2')
os.makedirs(target_dir, exist_ok=True)

# ===== 複製並解壓縮 Raw.zip =====
src_path = '/content/drive/MyDrive/datasets/CH-SIMS-V2/Raw.zip'
dst_path = os.path.join(target_dir, 'Raw.zip')

if not os.path.exists(src_path):
    raise FileNotFoundError(f"❌ 找不到來源檔案：{src_path}")

if not os.path.exists(dst_path):
    print(f"📥 從 Google Drive 複製 Raw.zip...")
    copy_with_progress(src_path, dst_path)
    print("✅ 複製完成！")
else:
    print(f"⏩ {dst_path} 已存在，跳過複製。")

if not os.path.exists(extract_dir):
    print(f"📂 解壓縮 Raw.zip...")
    with zipfile.ZipFile(dst_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file in tqdm(file_list, desc="解壓縮中"):
            zip_ref.extract(file, extract_dir)
    print("✅ 解壓縮完成！")
else:
    print(f"⏩ {extract_dir} 已存在，跳過解壓縮。")

# ===== 複製並解壓縮 bbox_cache =====
bbox_cache_name = "bbox_cache_merged"
bbox_cache_src = f'/content/drive/MyDrive/datasets/CH-SIMS-V2/{bbox_cache_name}.zip'
bbox_cache_local_zip = os.path.join(target_dir, f'{bbox_cache_name}.zip')
bbox_cache_dir = os.path.join(target_dir, 'CH-SIMS-V2', bbox_cache_name)

if os.path.exists(bbox_cache_src):
    if not os.path.exists(bbox_cache_local_zip):
        print(f"📥 複製 {bbox_cache_name}.zip...")
        shutil.copy2(bbox_cache_src, bbox_cache_local_zip)

    if not os.path.exists(bbox_cache_dir):
        print(f"📂 解壓縮 {bbox_cache_name}.zip...")
        with zipfile.ZipFile(bbox_cache_local_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(target_dir, 'CH-SIMS-V2'))
        print(f"✅ {bbox_cache_name} 解壓縮完成！")
else:
    bbox_cache_folder_src = '/content/drive/MyDrive/datasets/CH-SIMS-V2/bbox_cache'
    if os.path.exists(bbox_cache_folder_src):
        if not os.path.exists(bbox_cache_dir):
            print(f"📥 複製 {bbox_cache_name} 資料夾...")
            shutil.copytree(bbox_cache_folder_src, bbox_cache_dir)
            print("✅ 複製完成！")
    else:
        print(f"⚠️ 找不到 {bbox_cache_name}")
        bbox_cache_dir = None

# ===== 複製 checkpoint 到本地 =====
CKPT_BASE_DIR = "/content/drive/MyDrive/videomae_v2_valence_ckpts"
PHASE1_EXPERIMENT_NAME = "exp07_mixup_prob02_alpha02_dropout05_huber_delta02"
PHASE2_EXPERIMENT_NAME = "p2_exp03_multiclip_k4_batch4_accum8_nomixup"
PHASE1_CKPT_DIR = os.path.join(CKPT_BASE_DIR, PHASE1_EXPERIMENT_NAME)
PHASE2_CKPT_DIR = os.path.join(PHASE1_CKPT_DIR, "phase2_experiments", PHASE2_EXPERIMENT_NAME)
ckpt_src = os.path.join(PHASE2_CKPT_DIR, "phase2_best_E18.pt")
ckpt_local = '/content/phase2_best.pt'

if not os.path.exists(ckpt_local):
    print(f"📥 複製 checkpoint 到本地...")
    shutil.copy2(ckpt_src, ckpt_local)
    print("✅ checkpoint 複製完成！")
else:
    print(f"⏩ {ckpt_local} 已存在，跳過複製。")

print("\n" + "=" * 60)
print("資料準備完成！")
print("=" * 60)

# ==========================================
# Step 2. 匯入套件與基本設定
# ==========================================
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import AutoImageProcessor, AutoModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# ===== 固定亂數種子 =====
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device = {device}")

if device.type == "cuda":
    torch.cuda.empty_cache()
gc.collect()

# ==========================================
# Step 3. 路徑與參數設定
# ==========================================
# 資料路徑
FUSION_CSV_PATH = "/content/drive/MyDrive/CH-SIMS-V2_Fusion/fusion_csv/fusion.csv"
VIDEO_ROOT = "/content/datasets/CH-SIMS-V2/ch-simsv2s/Raw"
BBOX_CACHE_DIR = f"/content/datasets/CH-SIMS-V2/{bbox_cache_name}"
CKPT_PATH = "/content/phase2_best.pt"
BAD_JSON_PATH = "/content/drive/MyDrive/datasets/CH-SIMS-V2/bad_videos.json"
OUTPUT_CSV_PATH = "/content/drive/MyDrive/CH-SIMS-V2_Fusion/fusion_csv/fusion_Video.csv"

# VideoMAE V2 設定
HF_REPO = "OpenGVLab/VideoMAEv2-Base"

# Inference 參數（Multi-Clip）
LOAD_TIMEOUT = 30
NUM_FRAMES = 16
N_CLIPS = 4
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = False
USE_AMP = (device.type == "cuda")

# timeout 防護
MAX_TIMEOUTS = 80
MAX_OTHER_ERRORS = 120

print(f"\n{'=' * 60}")
print("Inference 設定 (VideoMAE V2 Multi-Clip)")
print(f"{'=' * 60}")
print(f"Fusion CSV: {FUSION_CSV_PATH}")
print(f"Video Root: {VIDEO_ROOT}")
print(f"Bbox Cache: {BBOX_CACHE_DIR}")
print(f"Checkpoint: {CKPT_PATH}")
print(f"Output CSV: {OUTPUT_CSV_PATH}")
print(f"Model: {HF_REPO}")
print(f"Num Frames per Clip: {NUM_FRAMES}")
print(f"Num Clips: {N_CLIPS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Mixed Precision: {USE_AMP}")
print(f"LOAD_TIMEOUT: {LOAD_TIMEOUT}s")
print(f"MAX_TIMEOUTS: {MAX_TIMEOUTS}")
print(f"MAX_OTHER_ERRORS: {MAX_OTHER_ERRORS}")
print(f"{'=' * 60}\n")

# ===== 讀取 bad_videos.json → BAD_IDX_SET =====
try:
    if os.path.exists(BAD_JSON_PATH):
        with open(BAD_JSON_PATH, "r", encoding="utf-8") as f:
            bad_data = json.load(f)
    else:
        bad_data = {"items": []}
except Exception:
    bad_data = {"items": []}

BAD_IDX_SET = {
    int(it["idx"]) for it in bad_data.get("items", [])
    if str(it.get("status", "")) != "ok" and "idx" in it
}
print(f"📋 已載入 {len(BAD_IDX_SET)} 個 bad idx（會直接跳過）")

# ===== bad_videos.json 追加寫入（讓下次直接跳過）=====
def append_bad_video(idx, video_id, clip_id, status, error, json_path=BAD_JSON_PATH):
    item = {
        "idx": int(idx),
        "video_id": str(video_id),
        "clip_id": str(clip_id),
        "status": str(status),
        "error": str(error)[:300]
    }

    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"items": []}
    except Exception:
        data = {"items": []}

    exists = any(int(x.get("idx", -1)) == int(idx) for x in data.get("items", []))
    if not exists:
        data["items"].append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==========================================
# Step 4. 載入 ImageProcessor
# ==========================================
print("🔁 載入 AutoImageProcessor...")
image_processor = AutoImageProcessor.from_pretrained(HF_REPO, trust_remote_code=True)
print("✅ ImageProcessor 載入完成")

# ==========================================
# Step 5. 定義 Dataset（Multi-Clip Inference 版本）
# ==========================================
class VideoMAEInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        video_root: str,
        image_processor,
        num_frames: int = 16,
        n_clips: int = 4,
        bbox_cache_dir: str = None,
        crop_scale: float = 1.3,
        target_size: tuple = (224, 224),
        bad_idx_set: set = None,
        max_timeouts: int = 80,
        max_other_errors: int = 120,
        load_timeout: int = 30,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.video_root = Path(video_root)
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.n_clips = n_clips
        self.bbox_cache_dir = Path(bbox_cache_dir) if bbox_cache_dir else None
        self.crop_scale = crop_scale
        self.target_size = target_size

        # 參數化的設定
        self.bad_idx_set = bad_idx_set or set()
        self.max_timeouts = max_timeouts
        self.max_other_errors = max_other_errors
        self.load_timeout = load_timeout

        # 計數器（instance 變數）
        self.timeout_counter = 0
        self.other_error_counter = 0

        # 記錄處理狀態
        self.detection_methods = {}
        self.failed_videos = []

        print(f"[Inference Dataset] samples = {len(self.df)}")
        print(f"[Inference Dataset] ASD crop enabled: {self.bbox_cache_dir is not None}")
        print(f"[Inference Dataset] Multi-clip: {self.n_clips} clips x {self.num_frames} frames")
        print(f"[Inference Dataset] Bad idx count: {len(self.bad_idx_set)}")
        print(f"[Inference Dataset] Load timeout: {self.load_timeout}s")

    def _load_bbox_cache(self, video_id: str, clip_id: str) -> dict:
        if self.bbox_cache_dir is None:
            return None
        cache_path = self.bbox_cache_dir / video_id / f"{clip_id}.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 無法讀取 bbox cache: {cache_path}, error: {e}")
            return None

    def _get_bbox_for_frame(self, frames_data: list, frame_idx: int) -> list:
        if not frames_data:
            return None
        frame_to_bbox = {f['frame']: f['bbox'] for f in frames_data}
        if frame_idx in frame_to_bbox:
            return frame_to_bbox[frame_idx]
        available_frames = sorted(frame_to_bbox.keys())
        closest_frame = min(available_frames, key=lambda x: abs(x - frame_idx))
        return frame_to_bbox[closest_frame]

    def _crop_frame(self, frame: np.ndarray, bbox: list, video_width: int, video_height: int) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        w_scaled = w * self.crop_scale
        h_scaled = h * self.crop_scale
        side = max(w_scaled, h_scaled)

        new_x1 = int(max(0, cx - side / 2))
        new_y1 = int(max(0, cy - side / 2))
        new_x2 = int(min(video_width, cx + side / 2))
        new_y2 = int(min(video_height, cy + side / 2))

        cropped = frame[new_y1:new_y2, new_x1:new_x2]
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return cv2.resize(frame, self.target_size)
        return cv2.resize(cropped, self.target_size)

    def _get_detection_method(self, bbox_cache: dict) -> str:
        if bbox_cache is None:
            return "no_cache"
        return bbox_cache.get('detection_method', 'unknown')

    def _get_face_data(self, bbox_cache: dict) -> tuple:
        if bbox_cache is None:
            return None, None
        if bbox_cache.get('active_speaker') and bbox_cache['active_speaker'].get('frames'):
            return bbox_cache['active_speaker']['frames'], 'active_speaker'
        if bbox_cache.get('fallback_face') and bbox_cache['fallback_face'].get('frames'):
            return bbox_cache['fallback_face']['frames'], 'fallback_face'
        return None, None

    def _uniform_sample(self, total_frames: int) -> np.ndarray:
        """均勻取樣（確定性，不使用隨機；不足時固定補最後一幀）"""
        if total_frames <= 0:
            return np.zeros(self.num_frames, dtype=int)
        if total_frames >= self.num_frames:
            return np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        base = np.arange(total_frames)
        pad = np.full(self.num_frames - total_frames, total_frames - 1, dtype=int)
        return np.concatenate([base, pad])

    def _multi_clip_sample(self, total_frames: int) -> np.ndarray:
        """
        Multi-clip: 均勻取 N 個片段，每個片段連續 num_frames 幀
        回傳 shape: (n_clips, num_frames)
        """
        if total_frames < self.num_frames:
            single_indices = self._uniform_sample(total_frames)
            return np.tile(single_indices, (self.n_clips, 1))

        all_indices = []
        clip_len = self.num_frames
        interval = total_frames / self.n_clips

        for i in range(self.n_clips):
            center_frame = int(interval * (i + 0.5))
            start_frame = center_frame - clip_len // 2
            if start_frame < 0:
                start_frame = 0
            elif start_frame + clip_len > total_frames:
                start_frame = total_frames - clip_len
            indices = np.arange(start_frame, start_frame + clip_len)
            all_indices.append(indices)

        return np.stack(all_indices, axis=0)

    def _load_video_frames_multiclip(self, video_path: Path, video_id: str, clip_id: str, idx: int) -> np.ndarray:
        """
        載入並處理影片 frames（Multi-Clip 版本）
        回傳 shape: (n_clips, num_frames, H, W, C)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        bbox_cache = self._load_bbox_cache(video_id, clip_id)
        self.detection_methods[idx] = self._get_detection_method(bbox_cache)

        face_frames, _face_source = self._get_face_data(bbox_cache)
        if face_frames is None:
            self.detection_methods[idx] = "no_face"
            raise ValueError(f"No face detected: {video_id}/{clip_id}")

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Empty video: {video_path}")

        if bbox_cache and bbox_cache.get('video_info'):
            video_width = bbox_cache['video_info']['width']
            video_height = bbox_cache['video_info']['height']
        else:
            first_frame = vr[0].asnumpy()
            video_height, video_width = first_frame.shape[:2]

        clip_indices = self._multi_clip_sample(total_frames)

        all_clips = []
        for clip_idx in range(self.n_clips):
            indices = clip_indices[clip_idx]
            raw_frames = vr.get_batch(indices).asnumpy()

            cropped_frames = []
            for i, frame_idx in enumerate(indices):
                frame = raw_frames[i]
                bbox = self._get_bbox_for_frame(face_frames, int(frame_idx))
                if bbox is not None:
                    cropped = self._crop_frame(frame, bbox, video_width, video_height)
                else:
                    cropped = cv2.resize(frame, self.target_size)
                cropped_frames.append(cropped)

            clip_frames = np.stack(cropped_frames, axis=0)
            all_clips.append(clip_frames)

        return np.stack(all_clips, axis=0)

    def _frames_to_pixel_values(self, clips_frames: np.ndarray) -> torch.Tensor:
        """將 numpy frames 轉換成 pixel_values tensor"""
        all_pixel_values = []
        for clip_idx in range(self.n_clips):
            frames = clips_frames[clip_idx]
            frames_list = [frames[i] for i in range(frames.shape[0])]
            inputs = self.image_processor(frames_list, return_tensors="pt")
            if "pixel_values" not in inputs:
                raise KeyError("找不到 pixel_values")
            pv = inputs["pixel_values"].squeeze(0)
            all_pixel_values.append(pv)
        return torch.stack(all_pixel_values, dim=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 檢查是否在 bad_idx_set 中
        if idx in self.bad_idx_set:
            self.detection_methods[idx] = "skip_bad_idx"
            return torch.zeros(self.n_clips, self.num_frames, 3, 224, 224), idx, False

        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        clip_id = str(row["clip_id"])
        video_path = self.video_root / video_id / f"{clip_id}.mp4"

        ex = None
        fut = None

        try:
            ex = ThreadPoolExecutor(max_workers=1)

            def _work():
                cf = self._load_video_frames_multiclip(video_path, video_id, clip_id, idx)
                pv = self._frames_to_pixel_values(cf)
                return pv

            fut = ex.submit(_work)
            pixel_values = fut.result(timeout=self.load_timeout)
            return pixel_values, idx, True

        except FuturesTimeoutError:
            self.timeout_counter += 1
            msg = f"decode_timeout (>{self.load_timeout}s)"
            print(f"⏰ Timeout loading {video_path} (idx={idx}) | total_timeouts={self.timeout_counter}", flush=True)

            self.detection_methods[idx] = "decode_timeout"
            self.failed_videos.append({
                "idx": idx, "video_id": video_id, "clip_id": clip_id, "error": "decode_timeout"
            })
            append_bad_video(idx, video_id, clip_id, "decode_timeout", msg)

            if self.timeout_counter >= self.max_timeouts:
                raise RuntimeError(
                    f"Too many timeouts: {self.timeout_counter} >= max_timeouts({self.max_timeouts}). "
                    "Stop to avoid slowdown."
                )

            return torch.zeros(self.n_clips, self.num_frames, 3, 224, 224), idx, False

        except Exception as e:
            self.other_error_counter += 1
            error_msg = str(e)
            if "No face detected" not in error_msg:
                print(f"⚠️ Error loading {video_path} (idx={idx}): {error_msg}", flush=True)

            self.failed_videos.append({
                "idx": idx, "video_id": video_id, "clip_id": clip_id, "error": error_msg
            })
            if idx not in self.detection_methods:
                self.detection_methods[idx] = "error"

            append_bad_video(idx, video_id, clip_id, "error", error_msg)

            if self.other_error_counter >= self.max_other_errors:
                raise RuntimeError(
                    f"Too many errors: {self.other_error_counter} >= max_other_errors({self.max_other_errors})."
                )

            return torch.zeros(self.n_clips, self.num_frames, 3, 224, 224), idx, False

        finally:
            if fut is not None:
                try:
                    fut.cancel()
                except Exception:
                    pass
            if ex is not None:
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    ex.shutdown(wait=False)


def collate_fn_inference(batch):
    pixels, indices, is_valid = zip(*batch)
    pixels = torch.stack(pixels, dim=0)
    return pixels, list(indices), list(is_valid)

# ==========================================
# Step 6. 定義 Regression Head
# ==========================================
class ValenceRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out = self.mlp(x)
        out = torch.tanh(out)
        return out.squeeze(-1)

# ==========================================
# Step 7. 載入模型與 checkpoint
# ==========================================
print("🔁 載入 VideoMAE V2 backbone...")
backbone = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True).to(device).eval()

with torch.no_grad():
    dummy_input = torch.zeros(1, 3, NUM_FRAMES, 224, 224).to(device)
    with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
        dummy_output = backbone(dummy_input)
        if isinstance(dummy_output, torch.Tensor):
            FEAT_DIM = dummy_output.shape[-1]
        else:
            FEAT_DIM = dummy_output.last_hidden_state.shape[-1]
    del dummy_input, dummy_output

print(f"✅ Backbone 載入完成，FEAT_DIM = {FEAT_DIM}")

head = ValenceRegressionHead(FEAT_DIM, hidden_dim=256, dropout=0.4).to(device).eval()

print(f"🔁 載入 checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)
backbone.load_state_dict(ckpt["backbone_state"])
head.load_state_dict(ckpt["head_state"])
print("✅ Checkpoint 載入完成")
print(f"   Phase: {ckpt.get('phase', 'N/A')}")
print(f"   Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"   Best Val CCC: {ckpt.get('best_val_ccc', 0.0):.4f}")
del ckpt

if device.type == "cuda":
    torch.cuda.empty_cache()
gc.collect()

# ==========================================
# Step 8. 讀取 fusion.csv 並準備 inference
# ==========================================
print(f"\n🔁 讀取 fusion.csv...")
df_fusion = pd.read_csv(FUSION_CSV_PATH)
print(f"✅ 共 {len(df_fusion)} 筆資料")
print(f"\n欄位: {list(df_fusion.columns)}")
print("\nmode 分佈:")
print(df_fusion["mode"].value_counts())

inference_dataset = VideoMAEInferenceDataset(
    df=df_fusion,
    video_root=VIDEO_ROOT,
    image_processor=image_processor,
    num_frames=NUM_FRAMES,
    n_clips=N_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR,
    crop_scale=1.3,
    target_size=(224, 224),
    bad_idx_set=BAD_IDX_SET,
    max_timeouts=MAX_TIMEOUTS,
    max_other_errors=MAX_OTHER_ERRORS,
    load_timeout=LOAD_TIMEOUT,
)

inference_loader = DataLoader(
    inference_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_inference,
    pin_memory=PIN_MEMORY,
)

print(f"\n總 batch 數: {len(inference_loader)}")

# ==========================================
# Step 9. 執行 Inference（Multi-Clip Mean Pooling）
# ==========================================
print("\n" + "=" * 60)
print("開始 Inference (Multi-Clip Mean Pooling)")
print("=" * 60)

all_predictions = {}
skipped_count = 0
start_time = time.time()
total_batches = len(inference_loader)

backbone.eval()
head.eval()

with torch.no_grad():
    for batch_idx, (pixel_values, indices, is_valid) in enumerate(tqdm(inference_loader, desc="Inference")):
        valid_indices = [i for i, v in enumerate(is_valid) if v]
        invalid_indices = [i for i, v in enumerate(is_valid) if not v]

        skipped_count += len(invalid_indices)
        if len(valid_indices) == 0:
            continue

        valid_pixels = pixel_values[valid_indices]
        valid_idx_list = [indices[i] for i in valid_indices]
        B_valid = valid_pixels.shape[0]

        # (B_valid*n_clips, C, T, H, W)
        valid_pixels_flat = (
            valid_pixels
            .view(-1, NUM_FRAMES, 3, 224, 224)
            .permute(0, 2, 1, 3, 4)
            .to(device, non_blocking=True)
        )

        with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            outputs = backbone(valid_pixels_flat)

            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 3:
                    if outputs.shape[1] == FEAT_DIM:
                        feats = outputs.mean(dim=2)
                    else:
                        feats = outputs.mean(dim=1)
                elif outputs.dim() == 2:
                    feats = outputs
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
            else:
                feats = outputs.last_hidden_state.mean(dim=1)

            preds_flat = head(feats)

        preds_flat = preds_flat.float().view(B_valid, N_CLIPS)
        preds = preds_flat.mean(dim=1).cpu().numpy()

        for i, idx0 in enumerate(valid_idx_list):
            all_predictions[idx0] = float(preds[i])

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB"
            print(f"  [{batch_idx+1}/{total_batches}] | "
                  f"Elapsed: {elapsed/60:.1f}min | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Skipped: {skipped_count}{gpu_mem}")

total_time = time.time() - start_time
print("\n✅ Inference 完成！")
print(f"   總耗時: {total_time/60:.1f} 分鐘")
print(f"   成功預測: {len(all_predictions)} 筆")
print(f"   跳過（timeout/錯誤/無臉等）: {skipped_count} 筆")

# ==========================================
# Step 10. 更新 fusion.csv
# ==========================================
print("\n🔁 更新 fusion.csv...")

# 使用不同的欄位名稱，避免覆蓋原有資料
df_fusion['pred_v_video'] = df_fusion.index.map(lambda idx: all_predictions.get(idx, np.nan))
df_fusion['has_face_video'] = df_fusion.index.map(lambda idx: inference_dataset.detection_methods.get(idx, 'unknown'))

pred_v_count = int(df_fusion['pred_v_video'].notna().sum())
pred_v_nan_count = int(df_fusion['pred_v_video'].isna().sum())
has_face_counts = df_fusion['has_face_video'].value_counts()

print("\n📊 更新統計:")
print(f"  pred_v_video 有預測值: {pred_v_count}/{len(df_fusion)}")
print(f"  pred_v_video 為 NaN: {pred_v_nan_count}/{len(df_fusion)}")
print("\n  has_face_video 分佈:")
for method, count in has_face_counts.items():
    print(f"    {method}: {count}")

if inference_dataset.failed_videos:
    no_face_videos = [f for f in inference_dataset.failed_videos if "No face detected" in f.get('error', '')]
    other_errors = [f for f in inference_dataset.failed_videos if "No face detected" not in f.get('error', '')]

    print("\n📋 跳過的影片統計:")
    print(f"  無人臉: {len(no_face_videos)} 個")
    print(f"  其他錯誤/timeout: {len(other_errors)} 個")

    if other_errors:
        print("\n⚠️ 其他錯誤/timeout 的影片 (前 10 筆):")
        for fail in other_errors[:10]:
            print(f"    {fail.get('video_id','?')}/{fail.get('clip_id','?')}: {str(fail.get('error',''))[:80]}")
        if len(other_errors) > 10:
            print(f"    ... 還有 {len(other_errors) - 10} 個")

# ==========================================
# Step 11. 儲存結果
# ==========================================
print(f"\n🔁 儲存結果到: {OUTPUT_CSV_PATH}")
df_fusion.to_csv(OUTPUT_CSV_PATH, index=False)
print("✅ 儲存完成！")

print("\n📋 結果預覽 (前 10 筆):")
preview_cols = ['video_id', 'clip_id', 'y_true', 'pred_v_video', 'has_face_video', 'mode']
print(df_fusion[preview_cols].head(10).to_string())

# ==========================================
# Step 12. 計算評估指標
# ==========================================
print("\n" + "=" * 60)
print("評估指標")
print("=" * 60)

def concordance_cc(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)

    mean_p = preds.mean()
    mean_t = targets.mean()
    var_p = preds.var()
    var_t = targets.var()

    vp = preds - mean_p
    vt = targets - mean_t
    corr = (vp * vt).mean() / (np.sqrt((vp**2).mean()) * np.sqrt((vt**2).mean()) + 1e-8)
    ccc = 2 * corr * np.sqrt(var_p * var_t) / (var_p + var_t + (mean_p - mean_t)**2 + 1e-8)
    return float(ccc)

def pearson_corr(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)

    vp = preds - preds.mean()
    vt = targets - targets.mean()
    pcc = (vp * vt).mean() / (np.sqrt((vp**2).mean()) * np.sqrt((vt**2).mean()) + 1e-8)
    return float(pcc)

def binary_metrics(preds, targets, threshold=0.0):
    preds = np.array(preds)
    targets = np.array(targets)

    preds_bin = (preds > threshold).astype(int)
    targets_bin = (targets > threshold).astype(int)

    acc = (preds_bin == targets_bin).mean()

    tp = ((preds_bin == 1) & (targets_bin == 1)).sum()
    fp = ((preds_bin == 1) & (targets_bin == 0)).sum()
    fn = ((preds_bin == 0) & (targets_bin == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(acc), float(f1)

# 使用新的欄位名稱
valid_mask = df_fusion['pred_v_video'].notna() & df_fusion['y_true'].notna()
df_valid = df_fusion[valid_mask]

if len(df_valid) > 0:
    preds = df_valid['pred_v_video'].values
    targets = df_valid['y_true'].values

    overall_ccc = concordance_cc(preds, targets)
    overall_pcc = pearson_corr(preds, targets)
    overall_acc, overall_f1 = binary_metrics(preds, targets)

    print(f"\n[Overall] ({len(df_valid)} samples)")
    print(f"  CCC: {overall_ccc:.4f}")
    print(f"  PCC: {overall_pcc:.4f}")
    print(f"  Acc: {overall_acc:.4f}")
    print(f"  F1:  {overall_f1:.4f}")

    for mode in ['train', 'valid', 'test']:
        df_mode = df_valid[df_valid['mode'] == mode]
        if len(df_mode) > 0:
            mode_preds = df_mode['pred_v_video'].values
            mode_targets = df_mode['y_true'].values

            mode_ccc = concordance_cc(mode_preds, mode_targets)
            mode_pcc = pearson_corr(mode_preds, mode_targets)
            mode_acc, mode_f1 = binary_metrics(mode_preds, mode_targets)

            print(f"\n[{mode}] ({len(df_mode)} samples)")
            print(f"  CCC: {mode_ccc:.4f}")
            print(f"  PCC: {mode_pcc:.4f}")
            print(f"  Acc: {mode_acc:.4f}")
            print(f"  F1:  {mode_f1:.4f}")

print("\n" + "=" * 60)
print("🎉 Inference 完成！")
print("=" * 60)
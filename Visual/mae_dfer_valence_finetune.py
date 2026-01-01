# ============================================
# MAE-DFER Valence Regression Fine-tuning
# Part 1: 環境設定與模型定義
# Single-Clip 版本 with ASD Fallback Face Cropping
# 適用於 Google Colab Pro (L4 GPU - 24GB VRAM)
# ============================================

# ============================================
# 0. 安裝套件（在 Colab 中執行）
# ============================================
"""
# 在 Colab 第一個 cell 執行：
!pip install -q timm==0.4.12 einops decord

from google.colab import drive
drive.mount('/content/drive')
"""

# ============================================
# 1. 匯入套件與基本設定
# ============================================
import os
import gc
import json
import math
import random
import time
from pathlib import Path
from functools import partial

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ---- 固定亂數種子 ----
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device =", device)

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()
gc.collect()

# ============================================
# 2. 路徑與超參數設定
# ============================================
# 🔧 請根據你的 Google Drive 路徑修改
CSV_PATH = "/content/drive/MyDrive/vjepa2_valence_ckpts/meta_filtered_182frames.csv"
VIDEO_ROOT = "/content/datasets/CH-SIMS-V2/ch-simsv2s/Raw"
BBOX_CACHE_DIR = "/content/drive/MyDrive/asd_bbox_cache"

# MAE-DFER 預訓練權重路徑（🔧 請修改為你的路徑）
PRETRAINED_PATH = "/content/drive/MyDrive/mae_dfer_weights/mae_dfer_pretrained.pth"

# ============================================
# 🔧 實驗管理
# ============================================
EXPERIMENT_NAME = "mae_dfer_valence_exp01_single_clip"

CKPT_BASE_DIR = "/content/drive/MyDrive/mae_dfer_valence_ckpts"
CKPT_DIR = os.path.join(CKPT_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_P1_LAST = os.path.join(CKPT_DIR, "phase1_last.pt")
CKPT_P1_BEST = os.path.join(CKPT_DIR, "phase1_best.pt")
CKPT_P2_LAST = os.path.join(CKPT_DIR, "phase2_last.pt")
CKPT_P2_BEST = os.path.join(CKPT_DIR, "phase2_best.pt")
BIN_THRESH_PATH = os.path.join(CKPT_DIR, "best_binary_threshold.txt")

# ===== MAE-DFER 模型設定 =====
NUM_FRAMES = 16
IMAGE_SIZE = 160          # MAE-DFER 使用 160×160
PATCH_SIZE = 16
TUBELET_SIZE = 2
FEAT_DIM = 512            # MAE-DFER hidden size
DEPTH = 16
NUM_HEADS = 8
LG_REGION_SIZE = (2, 5, 10)

# ===== 通用設定 =====
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
USE_AMP = (device.type == "cuda")
GRAD_CLIP_NORM = 1.0
DROPOUT_HEAD = 0.5

# ===== Phase 1 超參數（L4 優化）=====
BATCH_SIZE_P1 = 16
ACCUM_STEPS_P1 = 2
MAX_EPOCHS_P1 = 30
LR_HEAD_P1 = 5e-4
PATIENCE_P1 = 5

# ===== Phase 2 超參數（L4 優化）=====
BATCH_SIZE_P2 = 8
ACCUM_STEPS_P2 = 4
MAX_EPOCHS_P2 = 30
LR_HEAD_P2 = 5e-5
LR_BACKBONE_P2 = 5e-6
PATIENCE_P2 = 5
N_UNFREEZE_LAYERS = 4

# Loss 設定
HUBER_DELTA = 0.4
CCC_WEIGHT = 0.0

# Resume 設定
RESUME_PHASE1 = True
RESUME_PHASE2 = True
SKIP_PHASE1_IF_EXISTS = True

# 儲存實驗設定
experiment_config = {
    "experiment_name": EXPERIMENT_NAME,
    "model": "MAE-DFER (LGI-Former)",
    "num_frames": NUM_FRAMES,
    "image_size": IMAGE_SIZE,
    "feat_dim": FEAT_DIM,
    "depth": DEPTH,
    "num_heads": NUM_HEADS,
    "lg_region_size": LG_REGION_SIZE,
    "phase1": {"batch_size": BATCH_SIZE_P1, "lr_head": LR_HEAD_P1, "max_epochs": MAX_EPOCHS_P1},
    "phase2": {"batch_size": BATCH_SIZE_P2, "lr_head": LR_HEAD_P2, "lr_backbone": LR_BACKBONE_P2},
}

config_path = os.path.join(CKPT_DIR, "experiment_config.json")
with open(config_path, "w") as f:
    json.dump(experiment_config, f, indent=2)

print(f"\n{'='*60}")
print(f"🧪 實驗: {EXPERIMENT_NAME}")
print(f"模型: MAE-DFER (LGI-Former)")
print(f"輸入: {NUM_FRAMES} frames × {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"Hidden Dim: {FEAT_DIM}, Depth: {DEPTH}")
print(f"📁 Checkpoint: {CKPT_DIR}")
print(f"{'='*60}\n")


# ============================================
# 3. 下載 MAE-DFER 原始模型定義
# ============================================
import subprocess
import sys

MODELING_FILE = "/content/modeling_finetune.py"

print("\n📥 下載 MAE-DFER 原始模型定義...")

# 下載原始 repo 的 modeling_finetune.py
download_url = "https://raw.githubusercontent.com/sunlicai/MAE-DFER/master/modeling_finetune.py"

try:
    # 使用 wget 下載
    result = subprocess.run(
        ["wget", "-q", "-O", MODELING_FILE, download_url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"wget failed: {result.stderr}")
    print(f"✅ 已下載: {MODELING_FILE}")
except Exception as e:
    print(f"⚠️ wget 失敗，嘗試使用 urllib: {e}")
    import urllib.request
    urllib.request.urlretrieve(download_url, MODELING_FILE)
    print(f"✅ 已下載: {MODELING_FILE}")

# 將下載目錄加入 Python path
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

# Import 原始模型
from modeling_finetune import (
    vit_base_dim512_local_global_attn_depth16_region_size2510_patch16_160
)

print("✅ MAE-DFER 模型定義載入成功！")
# ============================================
# MAE-DFER Valence Regression Fine-tuning
# Part 2: Dataset 與 DataLoader
# ============================================

# ============================================
# 4. 讀取 CSV 並過濾無人臉影片
# ============================================
print("\n📂 讀取資料集...")
df = pd.read_csv(CSV_PATH)
print(f"meta.csv 總筆數: {len(df)}")

required_cols = {"video_id", "clip_id", "label_V", "mode"}
assert required_cols.issubset(df.columns), f"需要欄位：{required_cols}"

print("\n原始 mode 分佈：")
print(df["mode"].value_counts())

# 過濾沒有人臉的影片
print("\n🔍 過濾沒有人臉的影片...")

def has_valid_face_data(bbox_cache_path: Path) -> bool:
    if not bbox_cache_path.exists():
        return False
    try:
        with open(bbox_cache_path, 'r') as f:
            cache_data = json.load(f)
        if cache_data.get('active_speaker') and cache_data['active_speaker'].get('frames'):
            return True
        if cache_data.get('fallback_face') and cache_data['fallback_face'].get('frames'):
            return True
        return False
    except:
        return False

def get_video_total_frames(video_path: Path) -> int:
    if not video_path.exists():
        return 0
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        return len(vr)
    except:
        return 0

valid_rows = []
no_face_videos = []
too_short_videos = []
MIN_FRAMES = NUM_FRAMES

for _, row in df.iterrows():
    video_id = row['video_id']
    clip_id = row['clip_id']
    cache_path = Path(BBOX_CACHE_DIR) / video_id / f"{clip_id}.json"
    video_path = Path(VIDEO_ROOT) / str(video_id) / f"{clip_id}.mp4"

    if not has_valid_face_data(cache_path):
        no_face_videos.append(f"{video_id}/{clip_id}")
        continue

    total_frames = get_video_total_frames(video_path)
    if total_frames < MIN_FRAMES:
        too_short_videos.append(f"{video_id}/{clip_id}")
        continue

    valid_rows.append(row)

df_filtered = pd.DataFrame(valid_rows)
print(f"  原始：{len(df)} 筆")
print(f"  過濾後：{len(df_filtered)} 筆")
print(f"  移除（無人臉）：{len(no_face_videos)} 筆")
print(f"  移除（太短）：{len(too_short_videos)} 筆")

print("\n過濾後 mode 分佈：")
print(df_filtered["mode"].value_counts())

df = df_filtered


# ============================================
# 5.1 統計 Face Type 分佈（訓練前單獨統計）
# ============================================
def count_face_stats(df_subset, bbox_cache_dir):
    """訓練前單獨統計 face type 分佈（避免多 worker 問題）"""
    stats = {'asd': 0, 'fallback': 0, 'none': 0}
    bbox_cache_path = Path(bbox_cache_dir)
    
    for _, row in df_subset.iterrows():
        cache_path = bbox_cache_path / row['video_id'] / f"{row['clip_id']}.json"
        if not cache_path.exists():
            stats['none'] += 1
            continue
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            if cache.get('active_speaker', {}).get('frames'):
                stats['asd'] += 1
            elif cache.get('fallback_face', {}).get('frames'):
                stats['fallback'] += 1
            else:
                stats['none'] += 1
        except:
            stats['none'] += 1
    
    return stats

print("\n📊 統計 Face Type 分佈...")
train_face_stats = count_face_stats(df[df['mode'] == 'train'], BBOX_CACHE_DIR)
valid_face_stats = count_face_stats(df[df['mode'] == 'valid'], BBOX_CACHE_DIR)
test_face_stats = count_face_stats(df[df['mode'] == 'test'], BBOX_CACHE_DIR)

print(f"  Train: ASD={train_face_stats['asd']}, Fallback={train_face_stats['fallback']}, None={train_face_stats['none']}")
print(f"  Valid: ASD={valid_face_stats['asd']}, Fallback={valid_face_stats['fallback']}, None={valid_face_stats['none']}")
print(f"  Test:  ASD={test_face_stats['asd']}, Fallback={test_face_stats['fallback']}, None={test_face_stats['none']}")


# ============================================
# 5. Dataset 定義
# ============================================
class MAEDFERValenceDataset(Dataset):
    """
    MAE-DFER Single-Clip Dataset for Valence Regression
    - 輸入尺寸: 160×160
    - 採樣: 16 frames
    - 人臉裁切: ASD Fallback 機制
    """
    def __init__(self, df: pd.DataFrame, mode: str, video_root: str, num_frames: int = 16,
                 bbox_cache_dir: str = None, crop_scale: float = 1.3, 
                 target_size: tuple = (160, 160)):
        super().__init__()
        assert mode in ["train", "valid", "test"]
        self.df = df[df["mode"] == mode].reset_index(drop=True)
        self.mode = mode
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.bbox_cache_dir = Path(bbox_cache_dir) if bbox_cache_dir else None
        self.crop_scale = crop_scale
        self.target_size = target_size
        self.use_aug = (mode == "train")

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        print(f"[{mode}] samples = {len(self.df)}, augmentation = {self.use_aug}")

    def _load_bbox_cache(self, video_id: str, clip_id: str) -> dict:
        if self.bbox_cache_dir is None:
            return None
        cache_path = self.bbox_cache_dir / video_id / f"{clip_id}.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _get_face_data(self, bbox_cache: dict) -> tuple:
        if bbox_cache is None:
            return None, 'none'
        if bbox_cache.get('active_speaker') and bbox_cache['active_speaker'].get('frames'):
            return bbox_cache['active_speaker'], 'asd'
        if bbox_cache.get('fallback_face') and bbox_cache['fallback_face'].get('frames'):
            return bbox_cache['fallback_face'], 'fallback'
        return None, 'none'

    def _get_bbox_for_frame(self, face_data: dict, frame_idx: int) -> list:
        if face_data is None or 'frames' not in face_data:
            return None
        frames_data = face_data['frames']
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
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        side = max(w, h) * self.crop_scale
        new_x1 = int(max(0, cx - side / 2))
        new_y1 = int(max(0, cy - side / 2))
        new_x2 = int(min(video_width, cx + side / 2))
        new_y2 = int(min(video_height, cy + side / 2))
        cropped = frame[new_y1:new_y2, new_x1:new_x2]
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return cv2.resize(frame, self.target_size)
        return cv2.resize(cropped, self.target_size)

    def _sample_frame_indices(self, total_frames: int, num_frames: int) -> np.ndarray:
        if total_frames >= num_frames:
            if self.use_aug:
                max_start = total_frames - num_frames
                start_idx = np.random.randint(0, max_start + 1)
                indices = np.arange(start_idx, start_idx + num_frames)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            indices = np.arange(total_frames)
            pad_indices = np.full(num_frames - total_frames, total_frames - 1)
            indices = np.concatenate([indices, pad_indices])
        return indices

    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        # Horizontal flip
        if np.random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()

        # 暫時先關閉 brightness 和 contrast 的 augmentation
        # Brightness
        # if np.random.random() < 0.3:
        #     factor = np.random.uniform(0.8, 1.2)
        #     frames = np.clip(frames.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        # Contrast
        # if np.random.random() < 0.3:
        #     factor = np.random.uniform(0.8, 1.2)
        #     frames = frames.astype(np.float32)
        #     mean_val = frames.mean()
        #     frames = np.clip((frames - mean_val) * factor + mean_val, 0, 255).astype(np.uint8)
        
        return frames

    def _normalize(self, frames: np.ndarray) -> torch.Tensor:
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - self.mean) / self.std
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        return frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["video_id"]
        clip_id = row["clip_id"]
        label_v = float(row["label_V"])
        video_path = self.video_root / str(video_id) / f"{clip_id}.mp4"

        try:
            bbox_cache = self._load_bbox_cache(video_id, clip_id)
            face_data, face_type = self._get_face_data(bbox_cache)

            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)

            if bbox_cache and bbox_cache.get('video_info'):
                video_width = bbox_cache['video_info']['width']
                video_height = bbox_cache['video_info']['height']
            else:
                first_frame = vr[0].asnumpy()
                video_height, video_width = first_frame.shape[:2]

            indices = self._sample_frame_indices(total_frames, self.num_frames)
            frames_raw = vr.get_batch(indices).asnumpy()

            if face_data is not None:
                cropped_frames = []
                for i, frame in enumerate(frames_raw):
                    frame_idx = int(indices[i])
                    bbox = self._get_bbox_for_frame(face_data, frame_idx)
                    if bbox is not None:
                        cropped = self._crop_frame(frame, bbox, video_width, video_height)
                    else:
                        cropped = cv2.resize(frame, self.target_size)
                    cropped_frames.append(cropped)
                frames = np.stack(cropped_frames, axis=0)
            else:
                frames = np.array([cv2.resize(f, self.target_size) for f in frames_raw])

            if self.use_aug:
                frames = self._augment_frames(frames)

            pixel_values = self._normalize(frames)

        except Exception as e:
            print(f"⚠️ Error loading {video_path}: {e}")
            pixel_values = torch.zeros(3, self.num_frames, self.target_size[0], self.target_size[1])

        label = torch.tensor(label_v, dtype=torch.float32)
        return pixel_values, label


# ============================================
# 6. 建立 DataLoader
# ============================================
print("\n🔄 建立 Datasets...")

g = torch.Generator()
g.manual_seed(42)

train_dataset = MAEDFERValenceDataset(
    df, "train", VIDEO_ROOT, NUM_FRAMES,
    bbox_cache_dir=BBOX_CACHE_DIR, crop_scale=1.3, target_size=(IMAGE_SIZE, IMAGE_SIZE)
)
valid_dataset = MAEDFERValenceDataset(
    df, "valid", VIDEO_ROOT, NUM_FRAMES,
    bbox_cache_dir=BBOX_CACHE_DIR, crop_scale=1.3, target_size=(IMAGE_SIZE, IMAGE_SIZE)
)
test_dataset = MAEDFERValenceDataset(
    df, "test", VIDEO_ROOT, NUM_FRAMES,
    bbox_cache_dir=BBOX_CACHE_DIR, crop_scale=1.3, target_size=(IMAGE_SIZE, IMAGE_SIZE)
)

print(f"\n🔄 建立 Phase 1 DataLoader (batch_size={BATCH_SIZE_P1})...")
train_loader_p1 = DataLoader(
    train_dataset, batch_size=BATCH_SIZE_P1, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn, generator=g
)
valid_loader_p1 = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE_P1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn
)
print(f"Train batches: {len(train_loader_p1)}, Valid batches: {len(valid_loader_p1)}")

# 測試 DataLoader
print("\n⏱️ 測試 DataLoader...")
for i, (pixels, labels) in enumerate(train_loader_p1):
    print(f"Batch {i}: shape={pixels.shape}, dtype={pixels.dtype}")
    print(f"  Expected: [B, C=3, T={NUM_FRAMES}, H={IMAGE_SIZE}, W={IMAGE_SIZE}]")
    if i >= 0:
        break

print("\n✅ Part 2 載入完成：Dataset 與 DataLoader")
# ============================================
# MAE-DFER Valence Regression Fine-tuning
# Part 3: 模型建立、訓練與評估
# ============================================

# ============================================
# 7. 建立模型並載入預訓練權重
# ============================================
print("\n🔁 建立 MAE-DFER backbone (LGI-Former)...")

backbone = vit_base_dim512_local_global_attn_depth16_region_size2510_patch16_160(
    num_classes=0,
    all_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE,
    use_mean_pooling=True,
    drop_path_rate=0.1,
)

# 載入預訓練權重
print(f"\n📥 載入預訓練權重: {PRETRAINED_PATH}")
if os.path.exists(PRETRAINED_PATH):
    checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除 head 相關權重（我們用自己的 regression head）
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    
    # 載入權重並詳細檢查
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    
    print(f"\n📊 權重載入檢查:")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"\n  ⚠️ Missing keys (前10個):")
        for k in missing_keys[:10]:
            print(f"     - {k}")
        if len(missing_keys) > 10:
            print(f"     ... 還有 {len(missing_keys) - 10} 個")
    
    if unexpected_keys:
        print(f"\n  ⚠️ Unexpected keys (前10個):")
        for k in unexpected_keys[:10]:
            print(f"     - {k}")
        if len(unexpected_keys) > 10:
            print(f"     ... 還有 {len(unexpected_keys) - 10} 個")
    
    # 計算載入成功率
    total_params = len(list(backbone.state_dict().keys()))
    loaded_params = total_params - len(missing_keys)
    load_rate = loaded_params / total_params * 100
    print(f"\n  ✅ 權重載入成功率: {load_rate:.1f}% ({loaded_params}/{total_params})")
    
    if load_rate < 90:
        print("  ⚠️ 警告：載入率低於 90%，預訓練效果可能受影響！")
    else:
        print("  ✅ 預訓練權重載入成功！")
else:
    print(f"⚠️ 找不到預訓練權重，使用隨機初始化")

backbone.to(device)

# 測試 forward pass
print("\n🔍 測試模型 forward pass...")
with torch.no_grad():
    sample_input = torch.randn(2, 3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
        output = backbone(sample_input)
    print(f"  Input: {sample_input.shape} -> Output: {output.shape}")
    print(f"  Expected: [2, {FEAT_DIM}]")

if device.type == "cuda":
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ============================================
# 8. Regression Head
# ============================================
class ValenceRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
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

head = ValenceRegressionHead(FEAT_DIM, hidden_dim=256, dropout=DROPOUT_HEAD).to(device)
print(f"\nHead 參數量：{sum(p.numel() for p in head.parameters()):,}")


# ============================================
# 9. Loss 函數與評估指標
# ============================================
class HuberCCCLoss(nn.Module):
    def __init__(self, huber_delta: float = 0.4, ccc_weight: float = 0.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.ccc_weight = ccc_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(preds, targets)
        if self.ccc_weight > 0:
            mean_p, mean_t = preds.float().mean(), targets.float().mean()
            var_p, var_t = preds.float().var(unbiased=False), targets.float().var(unbiased=False)
            covar = ((preds.float() - mean_p) * (targets.float() - mean_t)).mean()
            ccc = (2 * covar) / (var_p + var_t + (mean_p - mean_t).pow(2) + 1e-8)
            return (1 - self.ccc_weight) * huber_loss + self.ccc_weight * (1 - ccc)
        return huber_loss

criterion = HuberCCCLoss(huber_delta=HUBER_DELTA, ccc_weight=CCC_WEIGHT)


def concordance_cc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = preds.detach().float(), targets.detach().float()
    mean_p, mean_t = preds.mean(), targets.mean()
    var_p, var_t = preds.var(unbiased=False), targets.var(unbiased=False)
    vp, vt = preds - mean_p, targets - mean_t
    corr = (vp * vt).mean() / (vp.pow(2).mean().sqrt() * vt.pow(2).mean().sqrt() + 1e-8)
    ccc = 2 * corr * torch.sqrt(var_p * var_t) / (var_p + var_t + (mean_p - mean_t).pow(2) + 1e-8)
    return float(ccc.item())


def pearson_corr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = preds.detach().float(), targets.detach().float()
    vp, vt = preds - preds.mean(), targets - targets.mean()
    return float(((vp * vt).mean() / (vp.pow(2).mean().sqrt() * vt.pow(2).mean().sqrt() + 1e-8)).item())


def mean_absolute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return float((preds - targets).abs().mean().item())


def binary_metrics_from_valence(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0):
    with torch.no_grad():
        preds_bin = (preds > threshold).long()
        targets_bin = (targets > threshold).long()
        correct = (preds_bin == targets_bin).float().mean().item()
        tp = ((preds_bin == 1) & (targets_bin == 1)).sum().item()
        fp = ((preds_bin == 1) & (targets_bin == 0)).sum().item()
        fn = ((preds_bin == 0) & (targets_bin == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(correct), float(f1)


# ============================================
# 10. 訓練迴圈
# ============================================
def run_one_epoch(backbone, head, data_loader, optimizer=None, scaler=None, train=True,
                  accum_steps: int = 1, use_amp: bool = True, bin_threshold: float = 0.0,
                  grad_clip_norm: float = None):
    if train:
        backbone.train()
        head.train()
    else:
        backbone.eval()
        head.eval()

    losses, all_preds, all_targets = [], [], []
    
    if train and optimizer is not None:
        optimizer.zero_grad()
        step_count = 0

    total_steps = len(data_loader)
    log_interval = max(1, total_steps // 10)
    epoch_start = time.time()

    for step, (pixel_values, labels) in enumerate(data_loader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                features = backbone(pixel_values)
                preds = head(features)
                raw_loss = criterion(preds, labels)

            if train and optimizer is not None:
                loss = raw_loss / accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_count += 1

                if step_count % accum_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if grad_clip_norm is not None:
                        params = list(backbone.parameters()) + list(head.parameters())
                        params = [p for p in params if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

        losses.append(raw_loss.item())
        all_preds.append(preds.float().detach().cpu())
        all_targets.append(labels.float().detach().cpu())

        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - epoch_start
            eta = (elapsed / (step + 1)) * (total_steps - step - 1)
            gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB" if device.type == "cuda" else ""
            print(f"  [{step+1:4d}/{total_steps}] loss={raw_loss.item():.4f} | ETA: {eta/60:.1f}min{gpu_mem}")

    # Handle remaining gradients
    if train and optimizer is not None and step_count % accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        if grad_clip_norm is not None:
            params = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = float(np.mean(losses))
    avg_ccc = concordance_cc(all_preds, all_targets)
    avg_pcc = pearson_corr(all_preds, all_targets)
    avg_mae = mean_absolute_error(all_preds, all_targets)
    avg_acc, avg_f1 = binary_metrics_from_valence(all_preds, all_targets, threshold=bin_threshold)

    return avg_loss, avg_ccc, avg_pcc, avg_mae, avg_acc, avg_f1


def collect_preds_targets(backbone, head, data_loader, use_amp: bool = True):
    backbone.eval()
    head.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for pixel_values, labels in data_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                features = backbone(pixel_values)
                preds = head(features)
            all_preds.append(preds.float().cpu())
            all_targets.append(labels.float().cpu())
    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


# ============================================
# 11. Phase 1：只訓練 Head
# ============================================
if SKIP_PHASE1_IF_EXISTS and os.path.exists(CKPT_P1_BEST):
    print("\n" + "=" * 60)
    print("✅ Phase 1 已存在，跳過")
    ckpt = torch.load(CKPT_P1_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    best_val_ccc_p1 = ckpt.get("best_val_ccc", -1e9)
    print(f"   Phase 1 best val CCC: {best_val_ccc_p1:.4f}")
else:
    print("\n" + "=" * 60)
    print("Phase 1: 只訓練 Head")
    print("=" * 60)

    for p in backbone.parameters():
        p.requires_grad = False

    optimizer_p1 = torch.optim.AdamW(head.parameters(), lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)
    scaler_p1 = GradScaler() if USE_AMP else None

    start_epoch_p1, best_val_ccc_p1, epochs_no_improve = 0, -1e9, 0

    # 初始化 history（先定義路徑）
    history_p1_path = os.path.join(CKPT_DIR, "history_p1.csv")
    history_p1 = []

    if RESUME_PHASE1 and os.path.exists(CKPT_P1_LAST):
        print(f"🔄 恢復 checkpoint...")
        ckpt = torch.load(CKPT_P1_LAST, map_location=device)
        backbone.load_state_dict(ckpt["backbone_state"])
        head.load_state_dict(ckpt["head_state"])
        optimizer_p1.load_state_dict(ckpt["optim_state"])
        if ckpt.get("scaler_state") and scaler_p1:
            scaler_p1.load_state_dict(ckpt["scaler_state"])
        start_epoch_p1 = ckpt.get("epoch", 0) + 1
        best_val_ccc_p1 = ckpt.get("best_val_ccc", -1e9)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        
        # 載入之前的 history
        if os.path.exists(history_p1_path):
            history_p1 = pd.read_csv(history_p1_path).to_dict('records')
            print(f"已載入 {len(history_p1)} 筆 Phase 1 訓練歷史")

    for epoch in range(start_epoch_p1, MAX_EPOCHS_P1):
        print(f"\n[Phase 1] Epoch {epoch+1}/{MAX_EPOCHS_P1}")

        print("Training...")
        train_loss, train_ccc, _, train_mae, _, _ = run_one_epoch(
            backbone, head, train_loader_p1, optimizer=optimizer_p1, scaler=scaler_p1,
            train=True, accum_steps=ACCUM_STEPS_P1, use_amp=USE_AMP, grad_clip_norm=GRAD_CLIP_NORM
        )

        print("Validating...")
        val_loss, val_ccc, _, val_mae, _, _ = run_one_epoch(
            backbone, head, valid_loader_p1, train=False, use_amp=USE_AMP
        )

        print(f"[Train] loss={train_loss:.4f}, CCC={train_ccc:.4f}, MAE={train_mae:.4f}")
        print(f"[Valid] loss={val_loss:.4f}, CCC={val_ccc:.4f}, MAE={val_mae:.4f}")

        ckpt_state = {
            "phase": 1, "epoch": epoch,
            "backbone_state": backbone.state_dict(), "head_state": head.state_dict(),
            "optim_state": optimizer_p1.state_dict(),
            "scaler_state": scaler_p1.state_dict() if scaler_p1 else None,
            "best_val_ccc": best_val_ccc_p1, "epochs_no_improve": epochs_no_improve,
        }
        torch.save(ckpt_state, CKPT_P1_LAST)

        history_p1.append({"epoch": epoch, "train_loss": train_loss, "train_ccc": train_ccc,
                          "val_loss": val_loss, "val_ccc": val_ccc})
        pd.DataFrame(history_p1).to_csv(history_p1_path, index=False)

        if val_ccc > best_val_ccc_p1:
            best_val_ccc_p1 = val_ccc
            epochs_no_improve = 0
            ckpt_state["best_val_ccc"] = best_val_ccc_p1
            torch.save(ckpt_state, CKPT_P1_BEST)
            print(f"✅ Best saved: {best_val_ccc_p1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement: {epochs_no_improve}/{PATIENCE_P1}")

        if epochs_no_improve >= PATIENCE_P1:
            print("⏹ Early stopping")
            break

    print(f"\n✅ Phase 1 完成，best val CCC = {best_val_ccc_p1:.4f}")


# ============================================
# 12. Phase 2：Fine-tune Backbone
# ============================================
print("\n" + "=" * 60)
print("Phase 2: Fine-tune backbone")
print("=" * 60)

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

# 載入 Phase 1 best
if os.path.exists(CKPT_P1_BEST):
    ckpt = torch.load(CKPT_P1_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])

# 凍結所有層，解凍最後 N 層
for p in backbone.parameters():
    p.requires_grad = False

total_layers = len(backbone.blocks)
print(f"Total layers: {total_layers}, Unfreezing last {N_UNFREEZE_LAYERS}")

for i in range(total_layers - N_UNFREEZE_LAYERS, total_layers):
    for p in backbone.blocks[i].parameters():
        p.requires_grad = True

for p in backbone.norm.parameters():
    p.requires_grad = True
if backbone.fc_norm is not None:
    for p in backbone.fc_norm.parameters():
        p.requires_grad = True

for p in head.parameters():
    p.requires_grad = True

backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Trainable: backbone={backbone_trainable:,}, head={head_trainable:,}")

# Phase 2 DataLoader
g_p2 = torch.Generator()
g_p2.manual_seed(42)

train_loader_p2 = DataLoader(
    train_dataset, batch_size=BATCH_SIZE_P2, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn, generator=g_p2
)
valid_loader_p2 = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE_P2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn
)

# Optimizer
backbone_params = [p for p in backbone.parameters() if p.requires_grad]
optimizer_p2 = torch.optim.AdamW([
    {"params": head.parameters(), "lr": LR_HEAD_P2},
    {"params": backbone_params, "lr": LR_BACKBONE_P2},
], weight_decay=WEIGHT_DECAY)
scaler_p2 = GradScaler() if USE_AMP else None

start_epoch_p2, best_val_ccc_p2, epochs_no_improve = 0, -1e9, 0

# 初始化 history（先定義路徑）
history_p2_path = os.path.join(CKPT_DIR, "history_p2.csv")
history_p2 = []

if RESUME_PHASE2 and os.path.exists(CKPT_P2_LAST):
    print(f"🔄 恢復 checkpoint...")
    ckpt2 = torch.load(CKPT_P2_LAST, map_location=device)
    backbone.load_state_dict(ckpt2["backbone_state"])
    head.load_state_dict(ckpt2["head_state"])
    optimizer_p2.load_state_dict(ckpt2["optim_state"])
    if ckpt2.get("scaler_state") and scaler_p2:
        scaler_p2.load_state_dict(ckpt2["scaler_state"])
    start_epoch_p2 = ckpt2.get("epoch", 0) + 1
    best_val_ccc_p2 = ckpt2.get("best_val_ccc", -1e9)
    epochs_no_improve = ckpt2.get("epochs_no_improve", 0)
    
    # 載入之前的 history
    if os.path.exists(history_p2_path):
        history_p2 = pd.read_csv(history_p2_path).to_dict('records')
        print(f"已載入 {len(history_p2)} 筆 Phase 2 訓練歷史")

for epoch in range(start_epoch_p2, MAX_EPOCHS_P2):
    print(f"\n[Phase 2] Epoch {epoch+1}/{MAX_EPOCHS_P2}")

    print("Training...")
    train_loss, train_ccc, _, train_mae, _, _ = run_one_epoch(
        backbone, head, train_loader_p2, optimizer=optimizer_p2, scaler=scaler_p2,
        train=True, accum_steps=ACCUM_STEPS_P2, use_amp=USE_AMP, grad_clip_norm=GRAD_CLIP_NORM
    )

    print("Validating...")
    val_loss, val_ccc, _, val_mae, _, _ = run_one_epoch(
        backbone, head, valid_loader_p2, train=False, use_amp=USE_AMP
    )

    print(f"[Train] loss={train_loss:.4f}, CCC={train_ccc:.4f}, MAE={train_mae:.4f}")
    print(f"[Valid] loss={val_loss:.4f}, CCC={val_ccc:.4f}, MAE={val_mae:.4f}")

    ckpt_state2 = {
        "phase": 2, "epoch": epoch,
        "backbone_state": backbone.state_dict(), "head_state": head.state_dict(),
        "optim_state": optimizer_p2.state_dict(),
        "scaler_state": scaler_p2.state_dict() if scaler_p2 else None,
        "best_val_ccc": best_val_ccc_p2, "epochs_no_improve": epochs_no_improve,
    }
    torch.save(ckpt_state2, CKPT_P2_LAST)

    history_p2.append({"epoch": epoch, "train_loss": train_loss, "train_ccc": train_ccc,
                       "val_loss": val_loss, "val_ccc": val_ccc})
    pd.DataFrame(history_p2).to_csv(history_p2_path, index=False)

    if val_ccc > best_val_ccc_p2:
        best_val_ccc_p2 = val_ccc
        epochs_no_improve = 0
        ckpt_state2["best_val_ccc"] = best_val_ccc_p2
        torch.save(ckpt_state2, CKPT_P2_BEST)
        print(f"✅ Best saved: {best_val_ccc_p2:.4f}")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement: {epochs_no_improve}/{PATIENCE_P2}")

    if epochs_no_improve >= PATIENCE_P2:
        print("⏹ Early stopping")
        break

print(f"\n✅ Phase 2 完成，best val CCC = {best_val_ccc_p2:.4f}")


# ============================================
# 13. 測試集評估
# ============================================
print("\n" + "=" * 60)
print("測試集評估")
print("=" * 60)

# 載入最佳模型
if os.path.exists(CKPT_P2_BEST):
    ckpt = torch.load(CKPT_P2_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    print("✅ 已載入 Phase 2 best checkpoint")

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE_P2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn
)

# 搜尋最佳 threshold
print("\n搜尋最佳 binary threshold...")
val_preds, val_targets = collect_preds_targets(backbone, head, valid_loader_p2, USE_AMP)

best_thr, best_f1 = 0.0, -1.0
for thr in np.linspace(-0.5, 0.5, 101):
    _, f1_thr = binary_metrics_from_valence(val_preds, val_targets, threshold=float(thr))
    if f1_thr > best_f1:
        best_f1, best_thr = f1_thr, float(thr)

print(f"最佳 threshold: {best_thr:.4f}, Valid F1: {best_f1:.4f}")

# 測試集評估
print("\n評估測試集...")
test_loss, test_ccc, test_pcc, test_mae, test_acc_zero, test_f1_zero = run_one_epoch(
    backbone, head, test_loader, train=False, use_amp=USE_AMP, bin_threshold=0.0
)

test_preds, test_targets = collect_preds_targets(backbone, head, test_loader, USE_AMP)
test_acc_best, test_f1_best = binary_metrics_from_valence(test_preds, test_targets, threshold=best_thr)

print(f"\n{'='*60}")
print(f"[Test Results - threshold=0.0]")
print(f"  Loss: {test_loss:.4f}")
print(f"  CCC:  {test_ccc:.4f}")
print(f"  PCC:  {test_pcc:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  Acc:  {test_acc_zero:.4f}")
print(f"  F1:   {test_f1_zero:.4f}")
print(f"\n[Test Results - best threshold={best_thr:.4f}]")
print(f"  Acc:  {test_acc_best:.4f}")
print(f"  F1:   {test_f1_best:.4f}")
print(f"{'='*60}")

# 儲存結果
summary_path = os.path.join(CKPT_DIR, "final_results.txt")
with open(summary_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write(f"MAE-DFER Valence Regression Results\n")
    f.write(f"Experiment: {EXPERIMENT_NAME}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Phase 1 best val CCC: {best_val_ccc_p1:.4f}\n")
    f.write(f"Phase 2 best val CCC: {best_val_ccc_p2:.4f}\n\n")
    f.write(f"[Test - threshold=0.0]\n")
    f.write(f"  CCC: {test_ccc:.4f}, PCC: {test_pcc:.4f}, MAE: {test_mae:.4f}\n")
    f.write(f"  Acc: {test_acc_zero:.4f}, F1: {test_f1_zero:.4f}\n\n")
    f.write(f"[Test - threshold={best_thr:.4f}]\n")
    f.write(f"  Acc: {test_acc_best:.4f}, F1: {test_f1_best:.4f}\n")

with open(BIN_THRESH_PATH, "w") as f:
    f.write(f"{best_thr:.6f}")

print(f"\n📄 結果已儲存至: {summary_path}")

# ============================================
# 14. 完成
# ============================================
print("\n" + "=" * 60)
print("🎉 訓練完成！")
print("=" * 60)
print(f"\n📁 輸出位置: {CKPT_DIR}")
print(f"  - phase1_best.pt / phase2_best.pt")
print(f"  - history_p1.csv / history_p2.csv")
print(f"  - final_results.txt")

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"\n🧹 GPU 記憶體已清理: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n✅ 全部完成！")
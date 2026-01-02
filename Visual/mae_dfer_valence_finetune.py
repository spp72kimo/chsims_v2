# ============================================
# MAE-DFER Valence Regression Fine-tuning
# Multi-Clip Sampling 版本
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
BBOX_CACHE_DIR = bbox_cache_dir  # 🔧 請修改為你的路徑

# MAE-DFER 預訓練權重路徑（🔧 請修改為你的路徑）
PRETRAINED_PATH = "/content/drive/MyDrive/Models/MAE-DFER/checkpoint-49.pth"

# ============================================
# 🔧 實驗管理（Phase 2 放在 Phase 1 底下）
# ============================================

# Phase 1 實驗名稱（通常固定，訓練好後不再改變）
PHASE1_EXPERIMENT_NAME = "mae_dfer_valence_p1_multiclip_k4"

# Phase 2 實驗名稱（🔧 每次跑不同參數時，只需要改這個名稱！）
PHASE2_EXPERIMENT_NAME = "p2_exp01_multiclip_k4_unfreeze4"

# ============================================

# Checkpoint 基礎目錄
CKPT_BASE_DIR = "/content/drive/MyDrive/mae_dfer_valence_ckpts"

# Phase 1 目錄
PHASE1_CKPT_DIR = os.path.join(CKPT_BASE_DIR, PHASE1_EXPERIMENT_NAME)
os.makedirs(PHASE1_CKPT_DIR, exist_ok=True)

# Phase 2 目錄（放在 Phase 1 底下）
PHASE2_CKPT_DIR = os.path.join(PHASE1_CKPT_DIR, "phase2_experiments", PHASE2_EXPERIMENT_NAME)
os.makedirs(PHASE2_CKPT_DIR, exist_ok=True)

# Phase 1 Checkpoints
CKPT_P1_LAST = os.path.join(PHASE1_CKPT_DIR, "phase1_last.pt")
CKPT_P1_BEST = os.path.join(PHASE1_CKPT_DIR, "phase1_best.pt")

# Phase 2 Checkpoints
CKPT_P2_LAST = os.path.join(PHASE2_CKPT_DIR, "phase2_last.pt")
CKPT_P2_BEST = os.path.join(PHASE2_CKPT_DIR, "phase2_best.pt")
BIN_THRESH_PATH = os.path.join(PHASE2_CKPT_DIR, "best_binary_threshold.txt")

# ===== MAE-DFER 模型設定 =====
NUM_FRAMES = 16           # 每個 clip 的 frame 數
IMAGE_SIZE = 160          # MAE-DFER 使用 160×160
PATCH_SIZE = 16
TUBELET_SIZE = 2
FEAT_DIM = 512            # MAE-DFER hidden size
DEPTH = 16
NUM_HEADS = 8
LG_REGION_SIZE = (2, 5, 10)

# ===== Multi-Clip 設定 =====
NUM_CLIPS = 4             # 🔧 分割成幾個 segments

# ===== 通用設定 =====
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
USE_AMP = (device.type == "cuda")
GRAD_CLIP_NORM = 1.0
DROPOUT_HEAD = 0.2

# ===== Phase 1 超參數（Multi-Clip 版本）=====
BATCH_SIZE_P1 = 4         # 🔧 降低（因為 multi-clip）
ACCUM_STEPS_P1 = 4        # 🔧 提高（維持 effective batch size = 16）
MAX_EPOCHS_P1 = 30
LR_HEAD_P1 = 5e-4
PATIENCE_P1 = 5

# ===== Phase 2 超參數（Multi-Clip 版本）=====
BATCH_SIZE_P2 = 2         # 🔧 降低（因為 multi-clip + fine-tune）
ACCUM_STEPS_P2 = 16       # 🔧 提高（維持 effective batch size = 32）
MAX_EPOCHS_P2 = 30
LR_HEAD_P2 = 5e-5
LR_BACKBONE_P2 = 5e-6
PATIENCE_P2 = 10
N_UNFREEZE_LAYERS = 4

# Loss 設定
HUBER_DELTA = 0.4
CCC_WEIGHT = 0.0

# Resume 設定
RESUME_PHASE1 = True
RESUME_PHASE2 = True
SKIP_PHASE1_IF_EXISTS = True

# ===== 最小 frames 要求 =====
MIN_FRAMES = NUM_CLIPS    # 至少要有 NUM_CLIPS 個 frames

# ===== 儲存 Phase 1 實驗設定 =====
phase1_config = {
    "experiment_name": PHASE1_EXPERIMENT_NAME,
    "model": "MAE-DFER (LGI-Former)",
    "num_frames": NUM_FRAMES,
    "num_clips": NUM_CLIPS,
    "image_size": IMAGE_SIZE,
    "feat_dim": FEAT_DIM,
    "depth": DEPTH,
    "num_heads": NUM_HEADS,
    "lg_region_size": LG_REGION_SIZE,
    "dropout_head": DROPOUT_HEAD,
    "batch_size": BATCH_SIZE_P1,
    "accum_steps": ACCUM_STEPS_P1,
    "effective_batch_size": BATCH_SIZE_P1 * ACCUM_STEPS_P1,
    "lr_head": LR_HEAD_P1,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "max_epochs": MAX_EPOCHS_P1,
    "patience": PATIENCE_P1,
}

phase1_config_path = os.path.join(PHASE1_CKPT_DIR, "experiment_config.json")
with open(phase1_config_path, "w") as f:
    json.dump(phase1_config, f, indent=2)

# ===== 儲存 Phase 2 實驗設定 =====
phase2_config = {
    "experiment_name": PHASE2_EXPERIMENT_NAME,
    "phase1_source": PHASE1_EXPERIMENT_NAME,
    "phase1_checkpoint": CKPT_P1_BEST,
    "model": "MAE-DFER (LGI-Former)",
    "num_frames": NUM_FRAMES,
    "num_clips": NUM_CLIPS,
    "image_size": IMAGE_SIZE,
    "feat_dim": FEAT_DIM,
    "dropout_head": DROPOUT_HEAD,
    "batch_size": BATCH_SIZE_P2,
    "accum_steps": ACCUM_STEPS_P2,
    "effective_batch_size": BATCH_SIZE_P2 * ACCUM_STEPS_P2,
    "lr_head": LR_HEAD_P2,
    "lr_backbone": LR_BACKBONE_P2,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "n_unfreeze_layers": N_UNFREEZE_LAYERS,
    "max_epochs": MAX_EPOCHS_P2,
    "patience": PATIENCE_P2,
    "huber_delta": HUBER_DELTA,
    "ccc_weight": CCC_WEIGHT,
}

phase2_config_path = os.path.join(PHASE2_CKPT_DIR, "experiment_config.json")
with open(phase2_config_path, "w") as f:
    json.dump(phase2_config, f, indent=2)

print(f"\n{'='*70}")
print(f"🧪 Phase 1 實驗名稱: {PHASE1_EXPERIMENT_NAME}")
print(f"🧪 Phase 2 實驗名稱: {PHASE2_EXPERIMENT_NAME}")
print(f"{'='*70}")
print(f"模型: MAE-DFER (LGI-Former)")
print(f"輸入: {NUM_CLIPS} clips × {NUM_FRAMES} frames × {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"Hidden Dim: {FEAT_DIM}, Depth: {DEPTH}")
print(f"\n📁 Phase 1 目錄: {PHASE1_CKPT_DIR}")
print(f"📁 Phase 2 目錄: {PHASE2_CKPT_DIR}")
print(f"\n🔧 Multi-Clip 設定:")
print(f"   Num Clips: {NUM_CLIPS}")
print(f"   Num Frames per Clip: {NUM_FRAMES}")
print(f"   P1 Batch/Accum: {BATCH_SIZE_P1}/{ACCUM_STEPS_P1} (effective: {BATCH_SIZE_P1 * ACCUM_STEPS_P1})")
print(f"   P2 Batch/Accum: {BATCH_SIZE_P2}/{ACCUM_STEPS_P2} (effective: {BATCH_SIZE_P2 * ACCUM_STEPS_P2})")
print(f"   P1 Forward Clips: {BATCH_SIZE_P1 * NUM_CLIPS}")
print(f"   P2 Forward Clips: {BATCH_SIZE_P2 * NUM_CLIPS}")
print(f"{'='*70}\n")


# ============================================
# 3. 下載並正確載入 MAE-DFER 模型
# ============================================
import subprocess
import sys

MODELING_FILE = "/content/modeling_finetune.py"

print("\n📥 下載 MAE-DFER 原始模型定義...")

download_url = "https://raw.githubusercontent.com/sunlicai/MAE-DFER/master/modeling_finetune.py"

try:
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

if "/content" not in sys.path:
    sys.path.insert(0, "/content")

# ============================================
# 🔧 關鍵修正：直接使用 VisionTransformer 類別建立 LGI-Former
# ============================================
import inspect
from modeling_finetune import VisionTransformer

def create_mae_dfer_lgi_former(
    img_size=160,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    embed_dim=512,
    depth=16,
    num_heads=8,
    lg_region_size=(2, 5, 10),
    num_classes=0,
    use_mean_pooling=True,
    drop_path_rate=0.1,
):
    """
    建立 MAE-DFER 的 LGI-Former 模型
    """
    actual_num_classes = 1 if num_classes == 0 else num_classes
    
    kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        use_mean_pooling=use_mean_pooling,
        num_classes=actual_num_classes,
        drop_path_rate=drop_path_rate,
        attn_type='local_global',
        lg_region_size=lg_region_size,
        lg_first_attn_type='cross',
        lg_third_attn_type='cross',
        lg_attn_param_sharing_first_third=False,
        lg_attn_param_sharing_all=False,
        lg_classify_token_type='org',
        lg_no_second=False,
        lg_no_third=False,
    )
    
    sig = inspect.signature(VisionTransformer.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard('self')
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    
    removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
    if removed:
        print(f"⚠️ 以下參數不被當前版本的 VisionTransformer 支援，已自動移除: {removed}")
    
    print(f"✅ 使用 {len(filtered_kwargs)} 個參數建立 LGI-Former")
    
    model = VisionTransformer(**filtered_kwargs)
    
    if num_classes == 0:
        model.head = nn.Identity()
        print(f"✅ 已將 head 替換為 Identity（作為 feature extractor 使用）")
    
    return model

print("✅ MAE-DFER LGI-Former 模型定義載入成功！")


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
    """檢查 bbox cache 是否有有效的人臉資料"""
    if not bbox_cache_path.exists():
        return False
    try:
        with open(bbox_cache_path, 'r') as f:
            cache_data = json.load(f)
        as_data = cache_data.get('active_speaker')
        if as_data and isinstance(as_data.get('frames'), list) and len(as_data['frames']) > 0:
            return True
        fb_data = cache_data.get('fallback_face')
        if fb_data and isinstance(fb_data.get('frames'), list) and len(fb_data['frames']) > 0:
            return True
        return False
    except:
        return False

def get_video_total_frames(video_path: Path, test_decode: bool = False) -> int:
    """取得影片總幀數"""
    if not video_path.exists():
        return 0
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return 0
        if test_decode:
            _ = vr[0].asnumpy()
            if total_frames > 1:
                _ = vr[total_frames - 1].asnumpy()
        return total_frames
    except Exception:
        return 0

valid_rows = []
no_face_videos = {
    'cache_missing': [],
    'cache_empty': [],
}
too_short_videos = []

for _, row in df.iterrows():
    video_id = str(row['video_id'])
    clip_id = row['clip_id']
    mode = row['mode']
    cache_path = Path(BBOX_CACHE_DIR) / video_id / f"{clip_id}.json"
    video_path = Path(VIDEO_ROOT) / video_id / f"{clip_id}.mp4"

    if not cache_path.exists():
        no_face_videos['cache_missing'].append(f"{video_id}/{clip_id} ({mode})")
        continue
    
    if not has_valid_face_data(cache_path):
        no_face_videos['cache_empty'].append(f"{video_id}/{clip_id} ({mode})")
        continue

    total_frames = get_video_total_frames(video_path, test_decode=False)
    if total_frames < MIN_FRAMES:
        too_short_videos.append(f"{video_id}/{clip_id} ({mode}): {total_frames} frames")
        continue

    valid_rows.append(row)

df_filtered = pd.DataFrame(valid_rows)

total_no_face = len(no_face_videos['cache_missing']) + len(no_face_videos['cache_empty'])
print(f"  原始：{len(df)} 筆")
print(f"  過濾後：{len(df_filtered)} 筆")
print(f"  移除（無人臉）：{total_no_face} 筆")
print(f"    - Cache 不存在：{len(no_face_videos['cache_missing'])} 筆")
print(f"    - Cache 空/無效：{len(no_face_videos['cache_empty'])} 筆")
print(f"  移除（太短 < {MIN_FRAMES} frames）：{len(too_short_videos)} 筆")

# 儲存沒有人臉的樣本清單
no_face_list_path = os.path.join(PHASE1_CKPT_DIR, "no_face_samples.txt")
with open(no_face_list_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("沒有人臉資料的樣本清單\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"[Cache 不存在] ({len(no_face_videos['cache_missing'])} 筆)\n")
    for item in no_face_videos['cache_missing']:
        f.write(f"  {item}\n")
    f.write(f"\n[Cache 空/無效] ({len(no_face_videos['cache_empty'])} 筆)\n")
    for item in no_face_videos['cache_empty']:
        f.write(f"  {item}\n")
    if too_short_videos:
        f.write(f"\n[太短] ({len(too_short_videos)} 筆)\n")
        for item in too_short_videos:
            f.write(f"  {item}\n")
print(f"\n📄 無人臉樣本清單已儲存: {no_face_list_path}")

print("\n過濾後 mode 分佈：")
print(df_filtered["mode"].value_counts())

df = df_filtered


# ============================================
# 5. Multi-Clip Dataset 定義
# ============================================
class MAEDFERMultiClipDataset(Dataset):
    """
    MAE-DFER Multi-Clip Dataset for Valence Regression
    
    將影片分割成 num_clips 個 segments，每個 segment 採樣 num_frames 個 frames
    - Train: 每個 segment 隨機採樣
    - Valid/Test: 每個 segment 中心採樣
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        video_root: str,
        num_frames: int = 16,
        num_clips: int = 4,
        bbox_cache_dir: str = None,
        crop_scale: float = 1.3,
        target_size: tuple = (160, 160),
    ):
        super().__init__()
        assert mode in ["train", "valid", "test"]
        self.df = df[df["mode"] == mode].reset_index(drop=True)
        self.mode = mode
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.bbox_cache_dir = Path(bbox_cache_dir) if bbox_cache_dir else None
        self.crop_scale = crop_scale
        self.target_size = target_size
        self.use_aug = (mode == "train")

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # 統計
        self.face_stats = {'asd': 0, 'fallback': 0, 'none': 0}
        self.short_videos = []
        self.skipped_videos = []

        print(f"[{mode}] samples = {len(self.df)}, clips = {num_clips}, frames/clip = {num_frames}")
        print(f"[{mode}] augmentation = {self.use_aug}")

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

    def _sample_segment_indices(self, seg_start: int, seg_end: int, num_frames: int, use_center: bool) -> np.ndarray:
        """
        從一個 segment 中採樣 num_frames 個 frames
        
        Args:
            seg_start: segment 起始 frame index
            seg_end: segment 結束 frame index (不包含)
            num_frames: 要採樣的 frame 數量
            use_center: True=中心採樣, False=隨機採樣
        
        Returns:
            採樣的 frame indices
        """
        seg_len = seg_end - seg_start

        if seg_len >= num_frames:
            if use_center:
                # 中心採樣
                center = seg_start + seg_len // 2
                start_idx = max(seg_start, center - num_frames // 2)
                end_idx = min(seg_end, start_idx + num_frames)
                if end_idx - start_idx < num_frames:
                    start_idx = seg_end - num_frames
                indices = np.arange(start_idx, start_idx + num_frames, dtype=int)
            else:
                # 隨機採樣
                max_start = seg_end - num_frames
                start_idx = np.random.randint(seg_start, max_start + 1)
                indices = np.arange(start_idx, start_idx + num_frames, dtype=int)
        else:
            # Segment 長度不足，使用 repeat padding
            base_indices = np.arange(seg_start, seg_end, dtype=int)
            num_pad = num_frames - seg_len
            pad_indices = np.full(num_pad, seg_end - 1, dtype=int)
            indices = np.concatenate([base_indices, pad_indices])

        return indices

    def _sample_multi_clips(self, vr: VideoReader, total_frames: int, use_center: bool) -> tuple:
        """
        將影片分割成 num_clips 個 segments，每個 segment 採樣 num_frames 個 frames
        
        Returns:
            tuple: (all_clips, all_indices)
        """
        actual_num_clips = min(self.num_clips, total_frames)

        # 計算每個 segment 的邊界
        segment_len = total_frames / actual_num_clips
        segments = []

        for i in range(actual_num_clips):
            seg_start = int(i * segment_len)
            seg_end = int((i + 1) * segment_len) if i < actual_num_clips - 1 else total_frames
            if seg_end <= seg_start:
                seg_end = seg_start + 1
            segments.append((seg_start, seg_end))

        # 從每個 segment 採樣
        all_clips = []
        all_indices = []

        for seg_start, seg_end in segments:
            indices = self._sample_segment_indices(seg_start, seg_end, self.num_frames, use_center)
            frames = vr.get_batch(indices).asnumpy()
            all_clips.append(frames)
            all_indices.append(indices)

        # 如果 actual_num_clips < num_clips，複製最後一個 clip 填充
        while len(all_clips) < self.num_clips:
            all_clips.append(all_clips[-1].copy())
            all_indices.append(all_indices[-1].copy())

        return all_clips, all_indices

    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """在 uint8 space 進行 augmentation"""
        # Horizontal flip
        if np.random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()
        return frames

    def _normalize(self, frames: np.ndarray) -> torch.Tensor:
        """Normalize 並轉換為 tensor"""
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - self.mean) / self.std
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        return frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, _retry_count=0):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        clip_id = row["clip_id"]
        label_v = float(row["label_V"])
        video_path = self.video_root / video_id / f"{clip_id}.mp4"

        try:
            bbox_cache = self._load_bbox_cache(video_id, clip_id)
            face_data, face_type = self._get_face_data(bbox_cache)

            # 統計 face type（只計數一次，以 sample 為單位）
            if idx < len(self.df):  # 確保只在第一次處理時計數
                pass  # 統計在外部處理

            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)

            if total_frames < MIN_FRAMES:
                raise ValueError(f"Video too short: {total_frames} < {MIN_FRAMES}")

            if bbox_cache and bbox_cache.get('video_info'):
                video_width = bbox_cache['video_info']['width']
                video_height = bbox_cache['video_info']['height']
            else:
                first_frame = vr[0].asnumpy()
                video_height, video_width = first_frame.shape[:2]

            # Multi-clip sampling
            use_center = not self.use_aug
            all_clips_raw, all_indices = self._sample_multi_clips(vr, total_frames, use_center)

            # Process each clip
            processed_clips = []
            for clip_frames, clip_indices in zip(all_clips_raw, all_indices):
                if face_data is not None:
                    cropped_frames = []
                    for frame, frame_idx in zip(clip_frames, clip_indices):
                        bbox = self._get_bbox_for_frame(face_data, int(frame_idx))
                        if bbox is not None:
                            cropped = self._crop_frame(frame, bbox, video_width, video_height)
                        else:
                            cropped = cv2.resize(frame, self.target_size)
                        cropped_frames.append(cropped)
                    clip_frames = np.stack(cropped_frames, axis=0)
                else:
                    clip_frames = np.array([cv2.resize(f, self.target_size) for f in clip_frames])

                # Augmentation (per clip)
                if self.use_aug:
                    clip_frames = self._augment_frames(clip_frames)

                # Normalize
                pixel_values = self._normalize(clip_frames)
                processed_clips.append(pixel_values)

            # Stack: [num_clips, C, T, H, W]
            multi_clip_pixels = torch.stack(processed_clips, dim=0)

        except Exception as e:
            if self.mode == "train":
                MAX_RETRIES = 10
                if _retry_count < MAX_RETRIES:
                    print(f"⚠️ [Train] 壞檔 {video_id}/{clip_id}，重抽樣本 (retry={_retry_count + 1})")
                    new_idx = np.random.randint(0, len(self.df))
                    return self.__getitem__(new_idx, _retry_count=_retry_count + 1)
                else:
                    raise RuntimeError(f"❌ 連續 {MAX_RETRIES} 次重抽都失敗！")
            else:
                print(f"⚠️ Error loading {video_path}: {e}")
                self.skipped_videos.append(str(video_path))
                multi_clip_pixels = torch.zeros(
                    self.num_clips, 3, self.num_frames,
                    self.target_size[0], self.target_size[1]
                )

        label = torch.tensor(label_v, dtype=torch.float32)
        return multi_clip_pixels, label

    def get_face_stats(self):
        return self.face_stats.copy()


def collate_fn_multi_clips(batch):
    """
    Collate function for multi-clip batches
    
    Input: list of (pixel_values, label)
        - pixel_values: [num_clips, C, T, H, W]
        - label: scalar
    
    Output: (batched_pixels, batched_labels)
        - batched_pixels: [B, num_clips, C, T, H, W]
        - batched_labels: [B]
    """
    pixels, labels = zip(*batch)
    pixels = torch.stack(pixels, dim=0)  # [B, num_clips, C, T, H, W]
    labels = torch.stack(labels, dim=0)  # [B]
    return pixels, labels


# ============================================
# 6. 統計 Face Type 分佈
# ============================================
def count_face_stats(df_subset, bbox_cache_dir):
    """統計 face type 分佈"""
    stats = {'asd': 0, 'fallback': 0, 'none': 0, 'cache_missing': 0}
    bbox_cache_path = Path(bbox_cache_dir)
    
    for _, row in df_subset.iterrows():
        cache_path = bbox_cache_path / str(row['video_id']) / f"{row['clip_id']}.json"
        if not cache_path.exists():
            stats['cache_missing'] += 1
            continue
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            as_data = cache.get('active_speaker')
            if as_data and isinstance(as_data.get('frames'), list) and len(as_data['frames']) > 0:
                stats['asd'] += 1
            else:
                fb_data = cache.get('fallback_face')
                if fb_data and isinstance(fb_data.get('frames'), list) and len(fb_data['frames']) > 0:
                    stats['fallback'] += 1
                else:
                    stats['none'] += 1
        except:
            stats['none'] += 1
    
    return stats

print("\n📊 統計 Face Type 分佈（過濾後）...")
train_face_stats = count_face_stats(df[df['mode'] == 'train'], BBOX_CACHE_DIR)
valid_face_stats = count_face_stats(df[df['mode'] == 'valid'], BBOX_CACHE_DIR)
test_face_stats = count_face_stats(df[df['mode'] == 'test'], BBOX_CACHE_DIR)

print(f"  Train: ASD={train_face_stats['asd']}, Fallback={train_face_stats['fallback']}")
print(f"  Valid: ASD={valid_face_stats['asd']}, Fallback={valid_face_stats['fallback']}")
print(f"  Test:  ASD={test_face_stats['asd']}, Fallback={test_face_stats['fallback']}")


# ============================================
# 7. 建立 Dataset 和 DataLoader
# ============================================
print("\n🔄 建立 Multi-Clip Datasets...")

g = torch.Generator()
g.manual_seed(42)

train_dataset = MAEDFERMultiClipDataset(
    df, "train", VIDEO_ROOT, NUM_FRAMES, NUM_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR, crop_scale=1.3, target_size=(IMAGE_SIZE, IMAGE_SIZE)
)
valid_dataset = MAEDFERMultiClipDataset(
    df, "valid", VIDEO_ROOT, NUM_FRAMES, NUM_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR, crop_scale=1.3, target_size=(IMAGE_SIZE, IMAGE_SIZE)
)
test_dataset = MAEDFERMultiClipDataset(
    df, "test", VIDEO_ROOT, NUM_FRAMES, NUM_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR, crop_scale=1.3, target_size=(IMAGE_SIZE, IMAGE_SIZE)
)

# 根據是否跳過 Phase 1 來決定是否建立 Phase 1 DataLoader
if not (SKIP_PHASE1_IF_EXISTS and os.path.exists(CKPT_P1_BEST)):
    print(f"\n🔄 建立 Phase 1 DataLoader (batch_size={BATCH_SIZE_P1})...")
    train_loader_p1 = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_P1, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, 
        collate_fn=collate_fn_multi_clips,
        worker_init_fn=worker_init_fn, generator=g
    )
    valid_loader_p1 = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE_P1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn_multi_clips,
        worker_init_fn=worker_init_fn
    )
    print(f"Train batches: {len(train_loader_p1)}, Valid batches: {len(valid_loader_p1)}")
    print(f"每次 Forward: {BATCH_SIZE_P1} × {NUM_CLIPS} = {BATCH_SIZE_P1 * NUM_CLIPS} clips")
else:
    print(f"\n⏩ 跳過建立 Phase 1 DataLoader（Phase 1 已存在）")
    train_loader_p1 = None
    valid_loader_p1 = None

# 測試 DataLoader
if train_loader_p1 is not None:
    print("\n⏱️ 測試 Multi-Clip DataLoader...")
    for i, (pixels, labels) in enumerate(train_loader_p1):
        print(f"Batch {i}: shape={pixels.shape}, dtype={pixels.dtype}")
        print(f"  Expected: [B={pixels.shape[0]}, K={pixels.shape[1]}, C={pixels.shape[2]}, T={pixels.shape[3]}, H={pixels.shape[4]}, W={pixels.shape[5]}]")
        if i >= 0:
            break

print("\n✅ Dataset 與 DataLoader 建立完成")


# ============================================
# 8. 建立模型並載入預訓練權重
# ============================================
print("\n🔁 建立 MAE-DFER backbone (LGI-Former)...")

backbone = create_mae_dfer_lgi_former(
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE,
    embed_dim=FEAT_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    lg_region_size=LG_REGION_SIZE,
    num_classes=0,
    use_mean_pooling=True,
    drop_path_rate=0.1,
)

# 載入預訓練權重
print(f"\n📥 載入預訓練權重: {PRETRAINED_PATH}")
if os.path.exists(PRETRAINED_PATH):
    checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"  原始權重數量: {len(state_dict)}")
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.') or k == 'mask_token':
            continue
        if k.startswith('encoder.'):
            new_key = k[len('encoder.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    state_dict = new_state_dict
    print(f"  處理後權重數量: {len(state_dict)}")
    
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    
    print(f"\n📊 權重載入檢查:")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    total_params = len(list(backbone.state_dict().keys()))
    loaded_params = total_params - len(missing_keys)
    load_rate = loaded_params / total_params * 100
    print(f"\n  ✅ 權重載入成功率: {load_rate:.1f}% ({loaded_params}/{total_params})")
else:
    print(f"⚠️ 找不到預訓練權重，使用隨機初始化")

backbone.to(device)

# 測試 forward pass（single clip）
print("\n🔍 測試模型 forward pass...")
with torch.no_grad():
    sample_input = torch.randn(2, 3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
        output = backbone(sample_input)
    print(f"  Single clip: Input {sample_input.shape} -> Output {output.shape}")
    print(f"  ✅ 模型已內建 pooling，輸出為 [B, {FEAT_DIM}]")

if device.type == "cuda":
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ============================================
# 9. Regression Head
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
# 10. Loss 函數與評估指標
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


def binary_metrics_from_valence(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0, weak_range: float = 0.4):
    """計算 Valence 的二元分類指標"""
    with torch.no_grad():
        preds = preds.detach().float()
        targets = targets.detach().float()
        
        preds_bin = (preds > threshold).long()
        targets_bin = (targets > threshold).long()
        
        acc2 = (preds_bin == targets_bin).float().mean().item()
        
        weak_mask = (targets.abs() <= weak_range)
        if weak_mask.sum() > 0:
            acc2_weak = (preds_bin[weak_mask] == targets_bin[weak_mask]).float().mean().item()
            n_weak = int(weak_mask.sum().item())
        else:
            acc2_weak = 0.0
            n_weak = 0
        
        tp = ((preds_bin == 1) & (targets_bin == 1)).sum().item()
        fp = ((preds_bin == 1) & (targets_bin == 0)).sum().item()
        fn = ((preds_bin == 0) & (targets_bin == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'acc2': float(acc2),
        'acc2_weak': float(acc2_weak),
        'n_weak_samples': n_weak,
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
    }


# ============================================
# 11. Multi-Clip 訓練迴圈
# ============================================
def run_one_epoch_multiclip(
    backbone, head, data_loader, optimizer=None, scaler=None, train=True,
    accum_steps: int = 1, use_amp: bool = True, bin_threshold: float = 0.0,
    grad_clip_norm: float = None
):
    """
    Multi-Clip 版本的訓練/驗證 loop
    
    輸入: [B, K, C, T, H, W]
    處理: Flatten 成 [B*K, C, T, H, W] 一起 forward
    聚合: 將 [B*K] 預測值 reshape 成 [B, K]，再 mean 成 [B]
    """
    if train:
        backbone.train()
        head.train()
    else:
        backbone.eval()
        head.eval()

    losses, all_preds, all_targets = [], [], []
    grad_norms = []

    if train and optimizer is not None:
        optimizer.zero_grad()
        step_count = 0

    total_steps = len(data_loader)
    log_interval = max(1, total_steps // 10)
    epoch_start = time.time()

    for step, (pixel_values, labels) in enumerate(data_loader):
        # pixel_values: [B, K, C, T, H, W]
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        B, K = pixel_values.shape[:2]

        # Flatten: [B*K, C, T, H, W]
        pixel_values_flat = pixel_values.view(B * K, *pixel_values.shape[2:])

        with torch.set_grad_enabled(train):
            with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # Forward all clips together
                feats = backbone(pixel_values_flat)  # [B*K, FEAT_DIM]
                
                # Get predictions for all clips: [B*K]
                preds_flat = head(feats)

                # Reshape and aggregate: [B*K] -> [B, K] -> [B]
                preds_per_clip = preds_flat.view(B, K)
                preds = preds_per_clip.mean(dim=1)  # Average across clips

                # Loss
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                        grad_norms.append(grad_norm.item())
                    
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
            grad_info = f" | grad: {np.mean(grad_norms[-10:]):.2f}" if grad_norms else ""
            print(f"  [{step+1:4d}/{total_steps}] loss={raw_loss.item():.4f} | clips={B*K}{grad_info} | ETA: {eta/60:.1f}min{gpu_mem}")

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
    bin_metrics = binary_metrics_from_valence(all_preds, all_targets, threshold=bin_threshold)
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

    metrics = {
        'loss': avg_loss,
        'ccc': avg_ccc,
        'pcc': avg_pcc,
        'mae': avg_mae,
        'acc2': bin_metrics['acc2'],
        'acc2_weak': bin_metrics['acc2_weak'],
        'f1': bin_metrics['f1'],
        'grad_norm': avg_grad_norm,
    }
    return metrics


def collect_preds_targets_multiclip(backbone, head, data_loader, use_amp: bool = True):
    """Multi-Clip 版本的預測收集"""
    backbone.eval()
    head.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for pixel_values, labels in data_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            B, K = pixel_values.shape[:2]
            pixel_values_flat = pixel_values.view(B * K, *pixel_values.shape[2:])

            with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                feats = backbone(pixel_values_flat)
                preds_flat = head(feats)
                preds_per_clip = preds_flat.view(B, K)
                preds = preds_per_clip.mean(dim=1)

            all_preds.append(preds.float().cpu())
            all_targets.append(labels.float().cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


# ============================================
# 12. Phase 1：只訓練 Head
# ============================================
if SKIP_PHASE1_IF_EXISTS and os.path.exists(CKPT_P1_BEST):
    print("\n" + "=" * 60)
    print("✅ Phase 1 已存在，跳過")
    ckpt = torch.load(CKPT_P1_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    best_val_ccc_p1 = ckpt.get("best_val_ccc", -1e9)
    print(f"   Phase 1 best val CCC: {best_val_ccc_p1:.4f}")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("Phase 1: 只訓練 Head (Multi-Clip)")
    print("=" * 60)
    print(f"  Batch Size: {BATCH_SIZE_P1}, Accum Steps: {ACCUM_STEPS_P1}")
    print(f"  Effective Batch Size: {BATCH_SIZE_P1 * ACCUM_STEPS_P1}")
    print(f"  Forward Clips per Batch: {BATCH_SIZE_P1 * NUM_CLIPS}")

    for p in backbone.parameters():
        p.requires_grad = False

    optimizer_p1 = torch.optim.AdamW(head.parameters(), lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)
    scaler_p1 = GradScaler() if USE_AMP else None

    start_epoch_p1, best_val_ccc_p1, epochs_no_improve = 0, -1e9, 0

    history_p1_path = os.path.join(PHASE1_CKPT_DIR, "history_p1.csv")
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
        
        if os.path.exists(history_p1_path):
            history_p1 = pd.read_csv(history_p1_path).to_dict('records')
            print(f"已載入 {len(history_p1)} 筆 Phase 1 訓練歷史")

    for epoch in range(start_epoch_p1, MAX_EPOCHS_P1):
        print(f"\n[Phase 1] Epoch {epoch+1}/{MAX_EPOCHS_P1}")

        print("Training...")
        train_metrics = run_one_epoch_multiclip(
            backbone, head, train_loader_p1, optimizer=optimizer_p1, scaler=scaler_p1,
            train=True, accum_steps=ACCUM_STEPS_P1, use_amp=USE_AMP, grad_clip_norm=GRAD_CLIP_NORM
        )

        print("Validating...")
        val_metrics = run_one_epoch_multiclip(
            backbone, head, valid_loader_p1, train=False, use_amp=USE_AMP
        )

        print(f"[Train] loss={train_metrics['loss']:.4f}, CCC={train_metrics['ccc']:.4f}, PCC={train_metrics['pcc']:.4f}, MAE={train_metrics['mae']:.4f}")
        print(f"        Acc2={train_metrics['acc2']:.4f}, Acc2_weak={train_metrics['acc2_weak']:.4f}, F1={train_metrics['f1']:.4f}")
        print(f"[Valid] loss={val_metrics['loss']:.4f}, CCC={val_metrics['ccc']:.4f}, PCC={val_metrics['pcc']:.4f}, MAE={val_metrics['mae']:.4f}")
        print(f"        Acc2={val_metrics['acc2']:.4f}, Acc2_weak={val_metrics['acc2_weak']:.4f}, F1={val_metrics['f1']:.4f}")

        ckpt_state = {
            "phase": 1, "epoch": epoch,
            "experiment_name": PHASE1_EXPERIMENT_NAME,
            "backbone_state": backbone.state_dict(), "head_state": head.state_dict(),
            "optim_state": optimizer_p1.state_dict(),
            "scaler_state": scaler_p1.state_dict() if scaler_p1 else None,
            "best_val_ccc": best_val_ccc_p1, "epochs_no_improve": epochs_no_improve,
        }
        torch.save(ckpt_state, CKPT_P1_LAST)

        history_p1.append({
            "epoch": epoch, 
            "train_loss": train_metrics['loss'], "train_ccc": train_metrics['ccc'], 
            "train_pcc": train_metrics['pcc'], "train_mae": train_metrics['mae'],
            "train_acc2": train_metrics['acc2'], "train_acc2_weak": train_metrics['acc2_weak'], "train_f1": train_metrics['f1'],
            "val_loss": val_metrics['loss'], "val_ccc": val_metrics['ccc'],
            "val_pcc": val_metrics['pcc'], "val_mae": val_metrics['mae'],
            "val_acc2": val_metrics['acc2'], "val_acc2_weak": val_metrics['acc2_weak'], "val_f1": val_metrics['f1'],
        })
        pd.DataFrame(history_p1).to_csv(history_p1_path, index=False)

        if val_metrics['ccc'] > best_val_ccc_p1:
            best_val_ccc_p1 = val_metrics['ccc']
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
# 13. Phase 2：Fine-tune Backbone
# ============================================
print("\n" + "=" * 70)
print(f"Phase 2: {PHASE2_EXPERIMENT_NAME}")
print(f"Fine-tune backbone (Multi-Clip)")
print("=" * 70)

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

if os.path.exists(CKPT_P1_BEST):
    ckpt = torch.load(CKPT_P1_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    print(f"✅ 已載入 Phase 1 best checkpoint")
    print(f"   Phase 1 val CCC: {ckpt.get('best_val_ccc', 'N/A')}")

for p in backbone.parameters():
    p.requires_grad = False

total_layers = len(backbone.blocks)
print(f"Total layers: {total_layers}, Unfreezing last {N_UNFREEZE_LAYERS}")

for i in range(total_layers - N_UNFREEZE_LAYERS, total_layers):
    for p in backbone.blocks[i].parameters():
        p.requires_grad = True

if hasattr(backbone, 'norm') and backbone.norm is not None and not isinstance(backbone.norm, nn.Identity):
    for p in backbone.norm.parameters():
        p.requires_grad = True

if hasattr(backbone, 'fc_norm') and backbone.fc_norm is not None and not isinstance(backbone.fc_norm, nn.Identity):
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

print(f"\n🔄 建立 Phase 2 DataLoader (batch_size={BATCH_SIZE_P2})...")
train_loader_p2 = DataLoader(
    train_dataset, batch_size=BATCH_SIZE_P2, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=collate_fn_multi_clips,
    worker_init_fn=worker_init_fn, generator=g_p2
)
valid_loader_p2 = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE_P2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=collate_fn_multi_clips,
    worker_init_fn=worker_init_fn
)
print(f"Train batches: {len(train_loader_p2)}, Valid batches: {len(valid_loader_p2)}")
print(f"每次 Forward: {BATCH_SIZE_P2} × {NUM_CLIPS} = {BATCH_SIZE_P2 * NUM_CLIPS} clips")

backbone_params = [p for p in backbone.parameters() if p.requires_grad]
optimizer_p2 = torch.optim.AdamW([
    {"params": head.parameters(), "lr": LR_HEAD_P2},
    {"params": backbone_params, "lr": LR_BACKBONE_P2},
], weight_decay=WEIGHT_DECAY)
scaler_p2 = GradScaler() if USE_AMP else None

start_epoch_p2, best_val_ccc_p2, epochs_no_improve = 0, -1e9, 0

history_p2_path = os.path.join(PHASE2_CKPT_DIR, "history_p2.csv")
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
    
    if os.path.exists(history_p2_path):
        history_p2 = pd.read_csv(history_p2_path).to_dict('records')
        print(f"已載入 {len(history_p2)} 筆 Phase 2 訓練歷史")

for epoch in range(start_epoch_p2, MAX_EPOCHS_P2):
    print(f"\n[Phase 2 - {PHASE2_EXPERIMENT_NAME}] Epoch {epoch+1}/{MAX_EPOCHS_P2}")

    print("Training...")
    train_metrics = run_one_epoch_multiclip(
        backbone, head, train_loader_p2, optimizer=optimizer_p2, scaler=scaler_p2,
        train=True, accum_steps=ACCUM_STEPS_P2, use_amp=USE_AMP, grad_clip_norm=GRAD_CLIP_NORM
    )

    print("Validating...")
    val_metrics = run_one_epoch_multiclip(
        backbone, head, valid_loader_p2, train=False, use_amp=USE_AMP
    )

    overfit_gap = train_metrics['ccc'] - val_metrics['ccc']

    print(f"\n{'='*70}")
    print(f"[Train] loss={train_metrics['loss']:.4f}, CCC={train_metrics['ccc']:.4f}, PCC={train_metrics['pcc']:.4f}, MAE={train_metrics['mae']:.4f}")
    print(f"        Acc2={train_metrics['acc2']:.4f}, Acc2_weak={train_metrics['acc2_weak']:.4f}, F1={train_metrics['f1']:.4f}")
    if train_metrics['grad_norm'] > 0:
        print(f"        grad_norm={train_metrics['grad_norm']:.3f}")
    print(f"[Valid] loss={val_metrics['loss']:.4f}, CCC={val_metrics['ccc']:.4f}, PCC={val_metrics['pcc']:.4f}, MAE={val_metrics['mae']:.4f}")
    print(f"        Acc2={val_metrics['acc2']:.4f}, Acc2_weak={val_metrics['acc2_weak']:.4f}, F1={val_metrics['f1']:.4f}")
    print(f"[Gap] Train-Valid CCC: {overfit_gap:.4f}")
    print(f"{'='*70}")

    ckpt_state2 = {
        "phase": 2, "epoch": epoch,
        "experiment_name": PHASE2_EXPERIMENT_NAME,
        "phase1_source": PHASE1_EXPERIMENT_NAME,
        "backbone_state": backbone.state_dict(), "head_state": head.state_dict(),
        "optim_state": optimizer_p2.state_dict(),
        "scaler_state": scaler_p2.state_dict() if scaler_p2 else None,
        "best_val_ccc": best_val_ccc_p2, "epochs_no_improve": epochs_no_improve,
    }
    torch.save(ckpt_state2, CKPT_P2_LAST)

    history_p2.append({
        "epoch": epoch, 
        "train_loss": train_metrics['loss'], "train_ccc": train_metrics['ccc'], 
        "train_pcc": train_metrics['pcc'], "train_mae": train_metrics['mae'],
        "train_acc2": train_metrics['acc2'], "train_acc2_weak": train_metrics['acc2_weak'], "train_f1": train_metrics['f1'],
        "train_grad_norm": train_metrics['grad_norm'],
        "val_loss": val_metrics['loss'], "val_ccc": val_metrics['ccc'],
        "val_pcc": val_metrics['pcc'], "val_mae": val_metrics['mae'],
        "val_acc2": val_metrics['acc2'], "val_acc2_weak": val_metrics['acc2_weak'], "val_f1": val_metrics['f1'],
        "overfit_gap": overfit_gap,
    })
    pd.DataFrame(history_p2).to_csv(history_p2_path, index=False)

    if val_metrics['ccc'] > best_val_ccc_p2:
        best_val_ccc_p2 = val_metrics['ccc']
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
# 14. 測試集評估
# ============================================
print("\n" + "=" * 60)
print("測試集評估")
print("=" * 60)

if os.path.exists(CKPT_P2_BEST):
    ckpt = torch.load(CKPT_P2_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    print("✅ 已載入 Phase 2 best checkpoint")

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE_P2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=collate_fn_multi_clips,
    worker_init_fn=worker_init_fn
)

print("\n搜尋最佳 binary threshold...")
val_preds, val_targets = collect_preds_targets_multiclip(backbone, head, valid_loader_p2, USE_AMP)

best_thr, best_f1 = 0.0, -1.0
for thr in np.linspace(-0.5, 0.5, 101):
    bin_metrics = binary_metrics_from_valence(val_preds, val_targets, threshold=float(thr))
    if bin_metrics['f1'] > best_f1:
        best_f1, best_thr = bin_metrics['f1'], float(thr)

print(f"最佳 threshold: {best_thr:.4f}, Valid F1: {best_f1:.4f}")

print("\n評估測試集...")
test_metrics = run_one_epoch_multiclip(
    backbone, head, test_loader, train=False, use_amp=USE_AMP, bin_threshold=0.0
)

test_preds, test_targets = collect_preds_targets_multiclip(backbone, head, test_loader, USE_AMP)
test_metrics_best_thr = binary_metrics_from_valence(test_preds, test_targets, threshold=best_thr)

print(f"\n{'='*60}")
print(f"[Test Results - threshold=0.0]")
print(f"  Loss:      {test_metrics['loss']:.4f}")
print(f"  CCC:       {test_metrics['ccc']:.4f}")
print(f"  PCC:       {test_metrics['pcc']:.4f}")
print(f"  MAE:       {test_metrics['mae']:.4f}")
print(f"  Acc2:      {test_metrics['acc2']:.4f}")
print(f"  Acc2_weak: {test_metrics['acc2_weak']:.4f}")
print(f"  F1:        {test_metrics['f1']:.4f}")
print(f"\n[Test Results - best threshold={best_thr:.4f}]")
print(f"  Acc2:      {test_metrics_best_thr['acc2']:.4f}")
print(f"  Acc2_weak: {test_metrics_best_thr['acc2_weak']:.4f}")
print(f"  F1:        {test_metrics_best_thr['f1']:.4f}")
print(f"{'='*60}")

summary_path = os.path.join(PHASE2_CKPT_DIR, "final_results.txt")
with open(summary_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write(f"MAE-DFER Valence Regression Results (Multi-Clip)\n")
    f.write(f"Phase 2 Experiment: {PHASE2_EXPERIMENT_NAME}\n")
    f.write("=" * 70 + "\n\n")
    f.write("=== Phase 1 來源 ===\n")
    f.write(f"實驗名稱: {PHASE1_EXPERIMENT_NAME}\n")
    f.write(f"Phase 1 best val CCC: {best_val_ccc_p1:.4f}\n\n")
    f.write("=== Phase 2 設定 ===\n")
    for k, v in phase2_config.items():
        f.write(f"{k}: {v}\n")
    f.write("\n=== 結果 ===\n")
    f.write(f"Phase 2 best val CCC: {best_val_ccc_p2:.4f}\n\n")
    f.write(f"Best binary threshold: {best_thr:.4f}\n\n")
    f.write("[Test Results - threshold=0.0]\n")
    f.write(f"  Loss:      {test_metrics['loss']:.4f}\n")
    f.write(f"  CCC:       {test_metrics['ccc']:.4f}\n")
    f.write(f"  PCC:       {test_metrics['pcc']:.4f}\n")
    f.write(f"  MAE:       {test_metrics['mae']:.4f}\n")
    f.write(f"  Acc2:      {test_metrics['acc2']:.4f}\n")
    f.write(f"  Acc2_weak: {test_metrics['acc2_weak']:.4f}\n")
    f.write(f"  F1:        {test_metrics['f1']:.4f}\n\n")
    f.write(f"[Test Results - threshold={best_thr:.4f}]\n")
    f.write(f"  Acc2:      {test_metrics_best_thr['acc2']:.4f}\n")
    f.write(f"  Acc2_weak: {test_metrics_best_thr['acc2_weak']:.4f}\n")
    f.write(f"  F1:        {test_metrics_best_thr['f1']:.4f}\n")

with open(BIN_THRESH_PATH, "w") as f:
    f.write(f"{best_thr:.6f}")

print(f"\n📄 結果已儲存至: {summary_path}")

print("\n" + "=" * 70)
print("🎉 Multi-Clip Training 完成！")
print("=" * 70)
print(f"\n📊 Multi-Clip 設定摘要：")
print(f"  Num Clips: {NUM_CLIPS}")
print(f"  Num Frames per Clip: {NUM_FRAMES}")
print(f"  Sampling: Train=Random, Valid/Test=Center")
print(f"  Aggregation: Prediction-level mean")
print(f"\n📁 輸出位置:")
print(f"  Phase 1: {PHASE1_CKPT_DIR}")
print(f"  Phase 2: {PHASE2_CKPT_DIR}")

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"\n🧹 GPU 記憶體已清理: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n✅ 全部完成！")
# ============================================
# VideoMAE V2 Valence Regression Training
# Multi-Clip Sampling 版本
# 使用 ASD Fallback 機制的情緒識別模型
# Google Colab 完整版 v5
# ============================================

# ============================================
# 0. 安裝套件 & 掛載 Google Drive（Colab 要先跑這段）
# ============================================
# !pip install -q "transformers[torch]" decord
#
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================
# 1. 匯入套件與基本設定
# ============================================
import os
import gc
import json
import math
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig

# ---- 固定亂數種子 ----
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    """
    DataLoader worker 初始化函數，確保每個 worker 有獨立但可重現的隨機種子

    使用公式: worker_seed = base_seed + worker_id
    這樣可以保證：
    1. 不同 worker 有不同的隨機序列（避免重複）
    2. 相同的 base_seed 可以重現結果
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device =", device)

# 清理 GPU 記憶體
if device.type == "cuda":
    torch.cuda.empty_cache()
gc.collect()

# ============================================
# 2. 路徑與超參數設定
# ============================================
CSV_PATH = "/content/drive/MyDrive/vjepa2_valence_ckpts/meta_filtered_182frames.csv"
VIDEO_ROOT = "/content/datasets/CH-SIMS-V2/ch-simsv2s/Raw"
BBOX_CACHE_DIR = bbox_cache_dir  # 🔧 請修改為你的路徑

# ============================================
# 🔧🔧🔧 實驗管理（Phase 2 放在 Phase 1 底下）🔧🔧🔧
# ============================================

# Phase 1 實驗名稱（通常固定，訓練好後不再改變）
PHASE1_EXPERIMENT_NAME = "exp07_mixup_prob02_alpha02_dropout05_huber_delta02"

# Phase 2 實驗名稱（🔧 每次跑不同參數時，只需要改這個名稱！）
PHASE2_EXPERIMENT_NAME = "p2_exp03_multiclip_k4_batch4_accum8_nomixup"

# ============================================

# Checkpoint 基礎目錄
CKPT_BASE_DIR = "/content/drive/MyDrive/videomae_v2_valence_ckpts"

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

# VideoMAE V2 checkpoint
HF_REPO = "OpenGVLab/VideoMAEv2-Base"

# ===== 模型設定 =====
NUM_FRAMES = 16           # VideoMAE V2 預設 16 frames（每個 clip）
IMAGE_SIZE = 224          # VideoMAE V2 預設 224x224
FEAT_DIM = 768            # ViT-B hidden size

# ===== Multi-Clip 設定 =====
NUM_CLIPS = 4             # 🔧 可調整：分割成幾個 segments

# ===== 通用設定 =====
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4

# Mixed Precision
USE_AMP = (device.type == "cuda")

# Gradient Clipping（避免梯度爆炸）
GRAD_CLIP_NORM = 1.0  # 🔧 可調整：None = 不使用 clipping

# ===== Head 設定 =====
DROPOUT_HEAD = 0.5        # Head 的 Dropout

# ===== Phase 1 超參數（通常固定） =====
BATCH_SIZE_P1   = 32
ACCUM_STEPS_P1  = 1
MAX_EPOCHS_P1   = 30
LR_HEAD_P1      = 5e-4
PATIENCE_P1     = 5
USE_MIXUP_P1    = True

# ============================================
# 🔧🔧🔧 Phase 2 超參數（Multi-Clip 版本）🔧🔧🔧
# ============================================
BATCH_SIZE_P2     = 4        # 🔧 降低 4 倍（因為 multi-clip）
ACCUM_STEPS_P2    = 8        # 🔧 提高 4 倍（維持 effective batch size）
MAX_EPOCHS_P2     = 30       # 🔧 可調整
LR_HEAD_P2        = 5e-5     # 🔧 可調整
LR_BACKBONE_P2    = 5e-6     # 🔧 可調整
PATIENCE_P2       = 5        # 🔧 可調整
N_UNFREEZE_LAYERS = 4        # 🔧 可調整：VideoMAE V2 ViT-B 共 12 層

# ⚠️ Multi-Clip 記憶體管理
# 實際 forward 的 clips 數 = BATCH_SIZE_P2 × NUM_CLIPS
# 例如: 4 × 4 = 16 clips（與原本 single-clip batch_size=16 相同）
#
# 如果遇到 OOM，可以：
# 1. 降低 BATCH_SIZE_P2（例如 2）並提高 ACCUM_STEPS_P2（例如 16）
# 2. 降低 NUM_CLIPS（例如 2）
# 3. 啟用 AUTO_ADJUST_BATCH_SIZE（自動調整）
AUTO_ADJUST_BATCH_SIZE = False  # 🔧 遇到 OOM 時自動減半 batch size

# Phase 2 資料增強設定（Multi-Clip 版本關閉 Mixup）
USE_MIXUP_P2 = False         # 🔧 Multi-clip 版本建議關閉
MIXUP_ALPHA  = 0.0
MIXUP_PROB   = 0.0

# Loss 設定
HUBER_DELTA = 0.4            # 🔧 可調整
CCC_WEIGHT = 0.0             # 🔧 Multi-clip 版本使用純 Huber loss

# Cutout（通常關閉）
USE_CUTOUT = False
# ============================================

# ===== Resume 設定 =====
RESUME_PHASE1 = True
RESUME_PHASE2 = True
SKIP_PHASE1_IF_EXISTS = True  # 設為 True 跳過 Phase 1（如果已訓練完成）

# ===== 儲存 Phase 1 實驗設定 =====
phase1_config = {
    "experiment_name": PHASE1_EXPERIMENT_NAME,
    "model": HF_REPO,
    "num_frames": NUM_FRAMES,
    "image_size": IMAGE_SIZE,
    "feat_dim": FEAT_DIM,
    "dropout_head": DROPOUT_HEAD,
    "batch_size": BATCH_SIZE_P1,
    "accum_steps": ACCUM_STEPS_P1,
    "effective_batch_size": BATCH_SIZE_P1 * ACCUM_STEPS_P1,
    "lr_head": LR_HEAD_P1,
    "weight_decay": WEIGHT_DECAY,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "max_epochs": MAX_EPOCHS_P1,
    "patience": PATIENCE_P1,
    "use_mixup": USE_MIXUP_P1,
}

phase1_config_path = os.path.join(PHASE1_CKPT_DIR, "experiment_config.json")
with open(phase1_config_path, "w") as f:
    json.dump(phase1_config, f, indent=2)

# ===== 儲存 Phase 2 實驗設定 =====
phase2_config = {
    "experiment_name": PHASE2_EXPERIMENT_NAME,
    "phase1_source": PHASE1_EXPERIMENT_NAME,
    "phase1_checkpoint": CKPT_P1_BEST,
    "model": HF_REPO,
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
    "use_mixup": USE_MIXUP_P2,
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
print(f"模型: {HF_REPO}")
print(f"輸入: {NUM_CLIPS} clips × {NUM_FRAMES} frames × {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"\n📁 Phase 1 目錄: {PHASE1_CKPT_DIR}")
print(f"📁 Phase 2 目錄: {PHASE2_CKPT_DIR}")
print(f"\n🔧 Phase 2 設定 (Multi-Clip):")
print(f"   Num Clips: {NUM_CLIPS}")
print(f"   Head Dropout: {DROPOUT_HEAD}")
print(f"   Batch Size: {BATCH_SIZE_P2} (effective: {BATCH_SIZE_P2 * ACCUM_STEPS_P2})")
print(f"   每次 Forward Clips 數量: {BATCH_SIZE_P2 * NUM_CLIPS}")
print(f"   LR Head: {LR_HEAD_P2}, LR Backbone: {LR_BACKBONE_P2}")
print(f"   Unfreeze Layers: {N_UNFREEZE_LAYERS}")
print(f"   Mixup: {USE_MIXUP_P2}")
print(f"   Huber Delta: {HUBER_DELTA}, CCC Weight: {CCC_WEIGHT}")
print(f"   Gradient Clipping: {GRAD_CLIP_NORM if GRAD_CLIP_NORM else 'Disabled'}")
print(f"\n⚠️ Multi-Clip 記憶體使用:")
print(f"   Phase 1: batch={BATCH_SIZE_P1} → {BATCH_SIZE_P1 * NUM_CLIPS} clips/forward")
print(f"   Phase 2: batch={BATCH_SIZE_P2} → {BATCH_SIZE_P2 * NUM_CLIPS} clips/forward (safer)")
print(f"   💡 如果 OOM，降低 BATCH_SIZE_P2 或 NUM_CLIPS")
print(f"{'='*70}\n")

# ============================================
# 3. 讀取 meta.csv 並過濾無人臉影片
# ============================================
df = pd.read_csv(CSV_PATH)
print("meta.csv 前幾列：")
print(df.head())

required_cols = {"video_id", "clip_id", "label_V", "mode"}
assert required_cols.issubset(df.columns), f"meta.csv 需要欄位：{required_cols}"

print("\n原始 mode 分佈：")
print(df["mode"].value_counts())

# ============================================
# 3.5 過濾沒有人臉的影片（支援 ASD Fallback）
# ============================================
print("\n🔍 過濾沒有人臉的影片...")

def has_valid_face_data(bbox_cache_path: Path) -> bool:
    """檢查是否有有效的人臉資料（支援 fallback）"""
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
    except Exception:
        return False

def get_video_total_frames(video_path: Path) -> int:
    """獲取影片總 frames 數"""
    if not video_path.exists():
        return 0
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        return len(vr)
    except Exception:
        return 0

valid_rows = []
no_face_videos = []
too_short_videos = []

# ✅ 最小 frames 要求（至少要能分成 num_clips 個 segments）
MIN_FRAMES = NUM_CLIPS  # 至少要有 num_clips 個 frames

for _, row in df.iterrows():
    video_id = row['video_id']
    clip_id = row['clip_id']
    cache_path = Path(BBOX_CACHE_DIR) / video_id / f"{clip_id}.json"
    video_path = Path(VIDEO_ROOT) / str(video_id) / f"{clip_id}.mp4"

    # 檢查人臉
    if not has_valid_face_data(cache_path):
        no_face_videos.append(f"{video_id}/{clip_id}")
        continue

    # ✅ 檢查影片長度
    total_frames = get_video_total_frames(video_path)
    if total_frames < MIN_FRAMES:
        too_short_videos.append({
            'video': f"{video_id}/{clip_id}",
            'frames': total_frames,
            'required': MIN_FRAMES
        })
        continue

    valid_rows.append(row)

df_filtered = pd.DataFrame(valid_rows)
print(f"  原始：{len(df)} 筆")
print(f"  過濾後：{len(df_filtered)} 筆")
print(f"  移除（無人臉）：{len(no_face_videos)} 筆")
print(f"  移除（太短）：{len(too_short_videos)} 筆 (< {MIN_FRAMES} frames)")

if no_face_videos:
    no_face_path = os.path.join(PHASE2_CKPT_DIR, "no_face_videos.txt")
    with open(no_face_path, "w", encoding="utf-8") as f:
        for v in sorted(no_face_videos):
            f.write(v + "\n")
    print(f"  無人臉影片清單已儲存至：{no_face_path}")

if too_short_videos:
    short_path = os.path.join(PHASE2_CKPT_DIR, "too_short_videos.txt")
    with open(short_path, "w", encoding="utf-8") as f:
        for v in too_short_videos:
            f.write(f"{v['video']}: {v['frames']} frames (required: {v['required']})\n")
    print(f"  太短影片清單已儲存至：{short_path}")
    if len(too_short_videos) <= 10:
        print(f"  範例：")
        for v in too_short_videos[:5]:
            print(f"    - {v['video']}: {v['frames']} frames")

print("\n過濾後 mode 分佈：")
print(df_filtered["mode"].value_counts())

df = df_filtered

# ============================================
# 4. 載入 VideoMAE V2 Processor
# ============================================
print("\n🔁 載入 VideoMAEImageProcessor...")
try:
    # ✅ 與 model 保持一致，加入 trust_remote_code=True
    video_processor = VideoMAEImageProcessor.from_pretrained(HF_REPO, trust_remote_code=True)
    print(f"VideoProcessor type: {type(video_processor)}")
except Exception as e:
    print(f"⚠️ 無法使用 VideoMAEImageProcessor，嘗試使用 AutoImageProcessor...")
    from transformers import AutoImageProcessor
    video_processor = AutoImageProcessor.from_pretrained(HF_REPO, trust_remote_code=True)
    print(f"VideoProcessor type: {type(video_processor)}")

# ✅ 驗證 processor 參數
print(f"\n📋 VideoProcessor 參數：")
if hasattr(video_processor, 'image_mean'):
    print(f"  Image Mean: {video_processor.image_mean}")
else:
    print(f"  Image Mean: (未設定，可能使用預設值)")

if hasattr(video_processor, 'image_std'):
    print(f"  Image Std: {video_processor.image_std}")
else:
    print(f"  Image Std: (未設定，可能使用預設值)")

if hasattr(video_processor, 'size'):
    print(f"  Size: {video_processor.size}")
elif hasattr(video_processor, 'crop_size'):
    print(f"  Crop Size: {video_processor.crop_size}")

if hasattr(video_processor, 'do_normalize'):
    print(f"  Do Normalize: {video_processor.do_normalize}")

if hasattr(video_processor, 'do_resize'):
    print(f"  Do Resize: {video_processor.do_resize}")

print(f"  Processor Class: {video_processor.__class__.__name__}")

# ============================================
# 5. 檢查 ASD bbox cache 覆蓋率
# ============================================
print("\n🔍 檢查 ASD bbox cache 統計...")

asd_stats = {
    'asd': 0,
    'single_face': 0,
    'center_face': 0,
    'no_face': 0,
    'no_cache': 0,
}

for _, row in df.iterrows():
    video_id = row['video_id']
    clip_id = row['clip_id']
    cache_path = Path(BBOX_CACHE_DIR) / video_id / f"{clip_id}.json"

    if not cache_path.exists():
        asd_stats['no_cache'] += 1
        continue

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        method = cache_data.get('detection_method', 'unknown')
        if method in asd_stats:
            asd_stats[method] += 1
        elif cache_data.get('active_speaker') and cache_data['active_speaker'].get('frames'):
            asd_stats['asd'] += 1
        elif cache_data.get('fallback_face') and cache_data['fallback_face'].get('frames'):
            asd_stats['single_face'] += 1
        else:
            asd_stats['no_face'] += 1
    except Exception:
        asd_stats['no_cache'] += 1

print(f"  總影片數：{len(df)}")
for method, count in asd_stats.items():
    pct = 100 * count / len(df) if len(df) > 0 else 0
    print(f"  {method}: {count} ({pct:.1f}%)")

# ============================================
# 6. 建立 Multi-Clip Video Dataset
# ============================================
class MultiClipVideoValenceDataset(Dataset):
    """
    Multi-Clip Video Dataset for Valence Regression

    將影片分割成 num_clips 個 segments，每個 segment 採樣 num_frames 個 frames
    - Train: 每個 segment 隨機採樣
    - Valid/Test: 每個 segment 中心採樣
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        video_root: str,
        video_processor,
        num_frames: int = 16,
        num_clips: int = 4,
        bbox_cache_dir: str = None,
        crop_scale: float = 1.3,
        target_size: tuple = (224, 224),
        clamp_augmented_values: bool = False,
        apply_aug_before_normalize: bool = True,  # 🔧 新增參數
    ):
        super().__init__()
        assert mode in ["train", "valid", "test"]
        self.df = df[df["mode"] == mode].reset_index(drop=True)
        self.mode = mode
        self.video_root = Path(video_root)
        self.video_processor = video_processor
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.skipped_videos = []

        self.bbox_cache_dir = Path(bbox_cache_dir) if bbox_cache_dir else None
        self.crop_scale = crop_scale
        self.target_size = target_size
        self.clamp_augmented_values = clamp_augmented_values
        self.apply_aug_before_normalize = apply_aug_before_normalize

        self.use_aug = (mode == "train")

        self.face_stats = {'asd': 0, 'fallback': 0, 'none': 0}
        self.short_videos = []  # 追蹤 total_frames < num_clips 的影片

        print(f"[{mode}] samples = {len(self.df)}, clips per sample = {num_clips}, augmentation = {self.use_aug}")
        print(f"[{mode}] Augmentation before normalize: {apply_aug_before_normalize}")
        print(f"[{mode}] Clamp augmented values: {clamp_augmented_values}")
        print(f"[{mode}] ASD fallback enabled: {self.bbox_cache_dir is not None}")

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

        resized = cv2.resize(cropped, self.target_size)
        return resized

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
                # 中心採樣：從 segment 中心向左右擴展
                center = seg_start + seg_len // 2
                start_idx = max(seg_start, center - num_frames // 2)
                end_idx = min(seg_end, start_idx + num_frames)

                # 如果接近邊界，調整範圍
                if end_idx - start_idx < num_frames:
                    start_idx = seg_end - num_frames

                indices = np.arange(start_idx, start_idx + num_frames, dtype=int)
            else:
                # 隨機採樣：在 segment 內隨機選擇起始點
                max_start = seg_end - num_frames
                start_idx = np.random.randint(seg_start, max_start + 1)
                indices = np.arange(start_idx, start_idx + num_frames, dtype=int)
        else:
            # Segment 長度不足，使用 repeat padding（重複最後一幀）
            base_indices = np.arange(seg_start, seg_end, dtype=int)
            num_pad = num_frames - seg_len
            # 重複最後一幀
            pad_indices = np.full(num_pad, seg_end - 1, dtype=int)
            indices = np.concatenate([base_indices, pad_indices])

        return indices

    def _sample_multi_clips(self, vr: VideoReader, total_frames: int, use_center: bool) -> tuple:
        """
        將影片分割成 num_clips 個 segments，每個 segment 採樣 num_frames 個 frames

        處理特殊情況：
        - total_frames < num_clips: 降級到 total_frames 個 clips（每個 segment 1 frame）
        - 某些 segments 長度為 0: 跳過該 segment

        Returns:
            tuple: (all_clips, all_indices)
                - all_clips: list of np.ndarray [actual_num_clips, num_frames, H, W, 3]
                - all_indices: list of np.ndarray [actual_num_clips, num_frames]
        """
        # ✅ 處理 total_frames < num_clips 的情況
        actual_num_clips = min(self.num_clips, total_frames)

        if actual_num_clips < self.num_clips:
            # 警告：影片太短，無法分割成指定數量的 clips
            # 這種情況很少見，但需要處理
            pass

        # 計算每個 segment 的邊界
        segment_len = total_frames / actual_num_clips
        segments = []

        for i in range(actual_num_clips):
            seg_start = int(i * segment_len)
            seg_end = int((i + 1) * segment_len) if i < actual_num_clips - 1 else total_frames

            # ✅ 確保 segment 至少有 1 frame
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

        # ✅ 如果 actual_num_clips < num_clips，需要 padding
        # 複製最後一個 clip 來填充
        while len(all_clips) < self.num_clips:
            all_clips.append(all_clips[-1].copy())
            all_indices.append(all_indices[-1].copy())

        return all_clips, all_indices

    def _augment_frames_uint8(self, frames: np.ndarray) -> np.ndarray:
        """
        在原始 uint8 image space 進行 augmentation（推薦）

        Args:
            frames: [T, H, W, 3] uint8 array, range [0, 255]

        Returns:
            augmented frames: [T, H, W, 3] uint8 array
        """
        # Horizontal flip
        if np.random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()

        # Brightness adjustment (在 [0, 255] space)
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = frames.astype(np.float32)
            frames = frames * brightness_factor
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        # Contrast adjustment (在 [0, 255] space)
        if np.random.random() < 0.3:
            contrast_factor = np.random.uniform(0.8, 1.2)
            frames = frames.astype(np.float32)
            mean_val = frames.mean()
            frames = (frames - mean_val) * contrast_factor + mean_val
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        return frames

    def _spatial_augment(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        在 normalized space 進行 augmentation（備用方案）

        Note: VideoMAE V2 使用 ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        標準化後理論範圍約 [-2.12, 2.64]

        ⚠️ 注意：在 normalized space 做 augmentation 與在 image space 做不等價
        建議使用 apply_aug_before_normalize=True
        """
        if np.random.random() < 0.5:
            pixel_values = torch.flip(pixel_values, dims=[-1])

        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            pixel_values = pixel_values * brightness_factor

        if np.random.random() < 0.3:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean_val = pixel_values.mean()
            pixel_values = (pixel_values - mean_val) * contrast_factor + mean_val

        # 🔧 可選的 clamp（預設關閉）
        if self.clamp_augmented_values:
            # 使用基於 processor normalization 的動態範圍
            if hasattr(self.video_processor, 'image_mean') and hasattr(self.video_processor, 'image_std'):
                mean = torch.tensor(self.video_processor.image_mean).view(3, 1, 1)
                std = torch.tensor(self.video_processor.image_std).view(3, 1, 1)

                # 計算理論範圍 + 5 std buffer (涵蓋 99.9999% 情況)
                min_val = ((0 - mean) / std).min() - 5
                max_val = ((1 - mean) / std).max() + 5

                pixel_values = pixel_values.clamp(min_val.item(), max_val.item())
            else:
                # Fallback: ImageNet 標準值的保守範圍
                pixel_values = pixel_values.clamp(-7, 8)

        return pixel_values

    def _load_multi_clips_with_crop(self, video_path: Path, video_id: str, clip_id: str) -> list:
        """
        載入影片並使用 multi-clip sampling + face cropping

        Returns:
            list of np.ndarray: [num_clips, num_frames, H, W, 3]
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        bbox_cache = self._load_bbox_cache(video_id, clip_id)

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        if total_frames <= 0:
            raise ValueError(f"Empty video: {video_path}")

        # ✅ 記錄短影片（total_frames < num_clips * num_frames）
        if total_frames < self.num_clips:
            self.short_videos.append({
                'video_id': video_id,
                'clip_id': clip_id,
                'total_frames': total_frames,
                'required_clips': self.num_clips
            })

        if bbox_cache and bbox_cache.get('video_info'):
            video_width = bbox_cache['video_info']['width']
            video_height = bbox_cache['video_info']['height']
        else:
            first_frame = vr[0].asnumpy()
            video_height, video_width = first_frame.shape[:2]

        # Multi-clip sampling - 返回 frames 和對應的實際 indices
        use_center = not self.use_aug
        all_clips_raw, all_indices = self._sample_multi_clips(vr, total_frames, use_center)

        # Face detection - ✅ 統計以 sample 為單位，只計數一次
        face_data, face_type = self._get_face_data(bbox_cache)
        if face_data is not None:
            self.face_stats[face_type] += 1
        else:
            self.face_stats['none'] += 1

        # Process each clip
        processed_clips = []
        for clip_frames, clip_indices in zip(all_clips_raw, all_indices):
            if face_data is not None:
                # ✅ 使用實際的 frame indices 來查找對應的 bbox
                cropped_frames = []
                for frame, frame_idx in zip(clip_frames, clip_indices):
                    # 使用實際的 frame index 查找 bbox
                    bbox = self._get_bbox_for_frame(face_data, int(frame_idx))

                    if bbox is not None:
                        cropped = self._crop_frame(frame, bbox, video_width, video_height)
                    else:
                        cropped = cv2.resize(frame, self.target_size)

                    cropped_frames.append(cropped)

                clip_frames = np.stack(cropped_frames, axis=0)
            else:
                # 沒有 face data，直接 resize（不重複計數）
                clip_frames = np.array([cv2.resize(f, self.target_size) for f in clip_frames])

            processed_clips.append(clip_frames)

        return processed_clips

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["video_id"]
        clip_id = row["clip_id"]
        label_v = float(row["label_V"])

        video_path = self.video_root / str(video_id) / f"{clip_id}.mp4"

        try:
            # Load multi-clips: list of [num_frames, H, W, 3]
            clips = self._load_multi_clips_with_crop(video_path, video_id, clip_id)

            # Process each clip with VideoProcessor
            all_pixel_values = []
            for clip_idx, clip_frames in enumerate(clips):
                # ✅ Augmentation 選項 1：在 processor 前（image space）
                if self.use_aug and self.apply_aug_before_normalize:
                    clip_frames = self._augment_frames_uint8(clip_frames)

                frames_list = [clip_frames[i] for i in range(clip_frames.shape[0])]
                inputs = self.video_processor(frames_list, return_tensors="pt")

                if "pixel_values" in inputs:
                    pixel_values = inputs["pixel_values"].squeeze(0)
                else:
                    raise KeyError("找不到 pixel_values")

                # ✅ 嚴格的維度檢查和轉換
                original_shape = pixel_values.shape

                # Case 1: [T, C, H, W] → [C, T, H, W]
                if pixel_values.dim() == 4 and pixel_values.shape[0] == self.num_frames:
                    pixel_values = pixel_values.permute(1, 0, 2, 3)

                # Case 2: [C, T, H, W] → 已經是正確格式
                elif pixel_values.dim() == 4 and pixel_values.shape[1] == self.num_frames:
                    pass  # 不需要 permute

                # Case 3: 其他格式 → 錯誤
                else:
                    raise ValueError(
                        f"Unexpected pixel_values shape from processor: {original_shape}\n"
                        f"Expected either [T={self.num_frames}, C=3, H, W] or [C=3, T={self.num_frames}, H, W]"
                    )

                # ✅ 最終驗證：確保格式正確
                assert pixel_values.dim() == 4, \
                    f"pixel_values should be 4D, got {pixel_values.dim()}D"
                assert pixel_values.shape[0] == 3, \
                    f"pixel_values should have C=3, got C={pixel_values.shape[0]}"
                assert pixel_values.shape[1] == self.num_frames, \
                    f"pixel_values should have T={self.num_frames}, got T={pixel_values.shape[1]}"

                # 第一個 clip 時顯示詳細資訊（debug 用）
                if clip_idx == 0 and idx < 3:  # 只在前幾個樣本顯示
                    print(f"  [Debug] Sample {idx}, Clip {clip_idx}: "
                          f"Processor output {original_shape} → Final {pixel_values.shape}")

                # ✅ Augmentation 選項 2：在 processor 後（normalized space）
                if self.use_aug and not self.apply_aug_before_normalize:
                    pixel_values = self._spatial_augment(pixel_values)

                all_pixel_values.append(pixel_values)

            # Stack: [num_clips, C, T, H, W]
            multi_clip_pixels = torch.stack(all_pixel_values, dim=0)

            # ✅ 最終 sanity check
            expected_shape = (self.num_clips, 3, self.num_frames, self.target_size[0], self.target_size[1])
            assert multi_clip_pixels.shape == expected_shape, \
                f"Final shape mismatch: got {multi_clip_pixels.shape}, expected {expected_shape}"

        except Exception as e:
            print(f"⚠️ Error loading {video_path} (idx={idx}): {e}")
            self.skipped_videos.append(str(video_path))
            multi_clip_pixels = torch.zeros(
                self.num_clips, 3, self.num_frames,
                self.target_size[0], self.target_size[1]
            )

        label = torch.tensor(label_v, dtype=torch.float32)
        return multi_clip_pixels, label

    def get_face_stats(self):
        return self.face_stats.copy()

    def get_short_videos_stats(self):
        """返回短影片統計"""
        return {
            'count': len(self.short_videos),
            'videos': self.short_videos.copy()
        }


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
# 7. 測試 Multi-Clip Dataset
# ============================================
print("\n🔍 測試 Multi-Clip Dataset 和 Processor 輸出格式...")
test_video_path = Path(VIDEO_ROOT) / df.iloc[0]["video_id"] / f"{df.iloc[0]['clip_id']}.mp4"
print(f"測試影片: {test_video_path}")

if test_video_path.exists():
    vr = VideoReader(str(test_video_path), ctx=cpu(0))
    total_frames = len(vr)
    print(f"影片總 frames: {total_frames}")
    print(f"分割成 {NUM_CLIPS} 個 segments，每個 segment 採樣 {NUM_FRAMES} frames")

    segment_len = total_frames / NUM_CLIPS
    for i in range(NUM_CLIPS):
        seg_start = int(i * segment_len)
        seg_end = int((i + 1) * segment_len) if i < NUM_CLIPS - 1 else total_frames
        print(f"  Segment {i+1}: frames [{seg_start}-{seg_end}) (length={seg_end-seg_start})")

    # ✅ 測試 processor 輸出格式
    print(f"\n🔍 測試 VideoProcessor 輸出格式...")
    indices = np.linspace(0, total_frames - 1, min(NUM_FRAMES, total_frames)).astype(int)
    test_frames = vr.get_batch(indices).asnumpy()
    test_frames_resized = [cv2.resize(test_frames[i], (IMAGE_SIZE, IMAGE_SIZE)) for i in range(len(test_frames))]

    test_inputs = video_processor(test_frames_resized, return_tensors="pt")

    print(f"  Input frames: {len(test_frames_resized)} frames × {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Processor output keys: {list(test_inputs.keys())}")

    if "pixel_values" in test_inputs:
        pixel_values = test_inputs["pixel_values"].squeeze(0)
        print(f"  Processor output shape (after squeeze): {pixel_values.shape}")

        # 檢測格式
        if pixel_values.dim() == 4:
            if pixel_values.shape[0] == NUM_FRAMES:
                print(f"  ✅ 格式: [T={pixel_values.shape[0]}, C={pixel_values.shape[1]}, H={pixel_values.shape[2]}, W={pixel_values.shape[3]}]")
                print(f"     需要 permute 成 [C, T, H, W]")
                pixel_values = pixel_values.permute(1, 0, 2, 3)
                print(f"  Permute 後: {pixel_values.shape}")
            elif pixel_values.shape[1] == NUM_FRAMES:
                print(f"  ✅ 格式: [C={pixel_values.shape[0]}, T={pixel_values.shape[1]}, H={pixel_values.shape[2]}, W={pixel_values.shape[3]}]")
                print(f"     已經是正確格式，不需要 permute")
            else:
                print(f"  ❌ 警告: 無法識別的格式 {pixel_values.shape}")
                print(f"     預期: [T={NUM_FRAMES}, C=3, H, W] 或 [C=3, T={NUM_FRAMES}, H, W]")

        # 最終驗證
        assert pixel_values.shape[0] == 3, f"預期 C=3, 得到 C={pixel_values.shape[0]}"
        assert pixel_values.shape[1] == NUM_FRAMES, f"預期 T={NUM_FRAMES}, 得到 T={pixel_values.shape[1]}"
        print(f"  ✅ 最終格式驗證通過: [C={pixel_values.shape[0]}, T={pixel_values.shape[1]}, H={pixel_values.shape[2]}, W={pixel_values.shape[3]}]")

        # 檢查數值範圍
        print(f"\n  數值範圍檢查:")
        print(f"    Min: {pixel_values.min().item():.4f}")
        print(f"    Max: {pixel_values.max().item():.4f}")
        print(f"    Mean: {pixel_values.mean().item():.4f}")
        print(f"    Std: {pixel_values.std().item():.4f}")

        # 期望範圍（ImageNet normalized）
        expected_min = -2.5  # 約 (0 - 0.485) / 0.225
        expected_max = 3.0   # 約 (1 - 0.406) / 0.225
        if pixel_values.min() < expected_min - 1 or pixel_values.max() > expected_max + 1:
            print(f"    ⚠️ 警告: 數值範圍異常，可能 normalization 有問題")
            print(f"    預期範圍: 約 [{expected_min:.1f}, {expected_max:.1f}]")
    else:
        print(f"  ❌ 錯誤: 找不到 'pixel_values' key")
else:
    print(f"⚠️ 測試影片不存在: {test_video_path}")

# ============================================
# 8. 建立 Dataset 和 DataLoader
# ============================================

# ============================================
# 8. 建立 Dataset 和 DataLoader
# ============================================

# 建立可重現的 generator
g = torch.Generator()
g.manual_seed(42)

# 🔧 CLAMP_AUGMENTED_VALUES: 是否對 augmented pixel values 進行 clamp
# False (推薦): 讓模型學習處理較大範圍的值，可能保留更多極端情緒表達
# True: 限制在理論範圍內，更保守但可能壓制極端值
CLAMP_AUGMENTED_VALUES = False

# 🔧 APPLY_AUG_BEFORE_NORMALIZE: Augmentation 的位置
# True (推薦): 在 image space (uint8 [0, 255]) 進行 augmentation，數學上正確
# False: 在 normalized space 進行 augmentation，較快但不等價於真正的 brightness/contrast
APPLY_AUG_BEFORE_NORMALIZE = True

print("\n🔄 建立 Datasets...")
train_dataset = MultiClipVideoValenceDataset(
    df, "train", VIDEO_ROOT, video_processor, NUM_FRAMES, NUM_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR,
    crop_scale=1.3,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    clamp_augmented_values=CLAMP_AUGMENTED_VALUES,
    apply_aug_before_normalize=APPLY_AUG_BEFORE_NORMALIZE,
)
valid_dataset = MultiClipVideoValenceDataset(
    df, "valid", VIDEO_ROOT, video_processor, NUM_FRAMES, NUM_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR,
    crop_scale=1.3,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    clamp_augmented_values=False,
    apply_aug_before_normalize=APPLY_AUG_BEFORE_NORMALIZE,
)
test_dataset = MultiClipVideoValenceDataset(
    df, "test", VIDEO_ROOT, video_processor, NUM_FRAMES, NUM_CLIPS,
    bbox_cache_dir=BBOX_CACHE_DIR,
    crop_scale=1.3,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    clamp_augmented_values=False,
    apply_aug_before_normalize=APPLY_AUG_BEFORE_NORMALIZE,
)

# ✅ 根據是否需要 Phase 1 來決定是否建立 Phase 1 DataLoader
if not (SKIP_PHASE1_IF_EXISTS and os.path.exists(CKPT_P1_BEST)):
    print(f"\n🔄 建立 Phase 1 DataLoader (batch_size={BATCH_SIZE_P1})...")
    train_loader_p1 = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_P1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_multi_clips,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    valid_loader_p1 = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE_P1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_multi_clips,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    print(f"Train batches: {len(train_loader_p1)}, Valid batches: {len(valid_loader_p1)}")
else:
    print(f"\n⏩ 跳過建立 Phase 1 DataLoader（將直接使用 Phase 2）")
    train_loader_p1 = None
    valid_loader_p1 = None

# ✅ Test loader 統一建立（但延遲到實際需要時）
# 先不建立，在測試階段再建立以節省記憶體
test_loader = None
print(f"Test DataLoader 將在測試階段建立")

# ============================================
# 9. 測試 DataLoader 速度
# ============================================
if train_loader_p1 is not None:
    print("\n⏱️ 測試 Multi-Clip DataLoader 速度...")
    start = time.time()
    for i, (pixels, labels) in enumerate(train_loader_p1):
        elapsed = time.time() - start
        print(f"Batch {i}: {elapsed:.2f}s, shape={pixels.shape}, dtype={pixels.dtype}")
        print(f"  Expected shape: [B={pixels.shape[0]}, K={pixels.shape[1]}, C={pixels.shape[2]}, T={pixels.shape[3]}, H={pixels.shape[4]}, W={pixels.shape[5]}]")
        if i >= 2:
            break
    print(f"DataLoader 預熱完成\n")

    print(f"\n📊 人臉偵測統計（預熱後）：")
    print(f"  Train: {train_dataset.get_face_stats()}")

    # ✅ 顯示短影片統計
    short_stats = train_dataset.get_short_videos_stats()
    if short_stats['count'] > 0:
        print(f"\n⚠️ 短影片警告：")
        print(f"  發現 {short_stats['count']} 個影片的 frames < {NUM_CLIPS} clips")
        print(f"  這些影片會使用 padding（複製最後一個 clip）")
        if short_stats['count'] <= 5:
            for v in short_stats['videos']:
                print(f"    - {v['video_id']}/{v['clip_id']}: {v['total_frames']} frames")
else:
    print("\n⏩ 跳過 DataLoader 測試（Phase 1 已存在）")

# ============================================
# 10. 載入 VideoMAE V2 backbone
# ============================================
print("🔁 載入 VideoMAE V2 backbone...")
config = AutoConfig.from_pretrained(HF_REPO, trust_remote_code=True)
backbone = AutoModel.from_pretrained(HF_REPO, config=config, trust_remote_code=True)
backbone.to(device)

# Test with multi-clip input
with torch.no_grad():
    sample_pixels, _ = train_dataset[0]
    # sample_pixels: [num_clips, C, T, H, W]

    # Test single clip first
    single_clip = sample_pixels[0:1].to(device)  # [1, C, T, H, W]
    print(f"Single clip input shape: {single_clip.shape}")

    with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
        outputs = backbone(pixel_values=single_clip)

        if isinstance(outputs, torch.Tensor):
            encoder_features = outputs
            print("📌 模型輸出為 Tensor")
        elif hasattr(outputs, 'last_hidden_state'):
            encoder_features = outputs.last_hidden_state
            print("📌 模型輸出為 BaseModelOutput")
        else:
            encoder_features = outputs[0] if isinstance(outputs, tuple) else outputs
            print(f"📌 模型輸出類型: {type(outputs)}")

        print(f"Encoder features shape (raw): {encoder_features.shape}")

        if encoder_features.dim() == 2:
            MODEL_ALREADY_POOLED = True
            actual_feat_dim = encoder_features.shape[-1]
            print("📌 模型已內建 pooling，輸出為 [B, hidden_dim]")
        elif encoder_features.dim() == 3:
            MODEL_ALREADY_POOLED = False
            actual_feat_dim = encoder_features.shape[-1]
            print("📌 模型輸出為 [B, num_patches, hidden_dim]，需要 mean pooling")
        else:
            raise ValueError(f"Unexpected output shape: {encoder_features.shape}")

    print(f"FEAT_DIM = {actual_feat_dim}")

    if actual_feat_dim != FEAT_DIM:
        print(f"⚠️ 更新 FEAT_DIM: {FEAT_DIM} -> {actual_feat_dim}")
        FEAT_DIM = actual_feat_dim

    if device.type == "cuda":
        print(f"GPU Memory after test forward: {torch.cuda.memory_allocated()/1e9:.2f} GB")

del sample_pixels, outputs, encoder_features, single_clip
if device.type == "cuda":
    torch.cuda.empty_cache()

print(f"\n📌 MODEL_ALREADY_POOLED = {MODEL_ALREADY_POOLED}")

# ============================================
# 11. 定義 MLP Regression Head
# ============================================
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

head = ValenceRegressionHead(FEAT_DIM, hidden_dim=256, dropout=DROPOUT_HEAD).to(device)

print(f"\nHead 結構：")
print(head)
print(f"Head 參數量：{sum(p.numel() for p in head.parameters()):,}")
print(f"Head Dropout: {DROPOUT_HEAD}")

# ============================================
# 12. Loss Functions
# ============================================
class HuberCCCLoss(nn.Module):
    """結合 Huber Loss 和 CCC Loss"""
    def __init__(self, huber_delta: float = 0.2, ccc_weight: float = 0.5):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.ccc_weight = ccc_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(preds, targets)

        if self.ccc_weight > 0:
            # CCC Loss
            preds_f = preds.float()
            targets_f = targets.float()

            mean_p = preds_f.mean()
            mean_t = targets_f.mean()
            var_p = preds_f.var(unbiased=False)
            var_t = targets_f.var(unbiased=False)

            vp = preds_f - mean_p
            vt = targets_f - mean_t
            covar = (vp * vt).mean()

            ccc = (2 * covar) / (var_p + var_t + (mean_p - mean_t).pow(2) + 1e-8)
            ccc_loss = 1 - ccc

            total_loss = (1 - self.ccc_weight) * huber_loss + self.ccc_weight * ccc_loss
        else:
            total_loss = huber_loss

        return total_loss

criterion = HuberCCCLoss(huber_delta=HUBER_DELTA, ccc_weight=CCC_WEIGHT)


def concordance_cc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    計算 Concordance Correlation Coefficient (CCC)

    Note: 使用 unbiased=False 以與 loss 中的 CCC 計算保持一致
    """
    preds = preds.detach().float()
    targets = targets.detach().float()

    mean_p = preds.mean()
    mean_t = targets.mean()
    var_p = preds.var(unbiased=False)  # ✅ 與 loss 保持一致
    var_t = targets.var(unbiased=False)  # ✅ 與 loss 保持一致

    vp = preds - mean_p
    vt = targets - mean_t
    corr = (vp * vt).mean() / (vp.pow(2).mean().sqrt() * vt.pow(2).mean().sqrt() + 1e-8)

    ccc = 2 * corr * torch.sqrt(var_p * var_t) / (var_p + var_t + (mean_p - mean_t).pow(2) + 1e-8)
    return float(ccc.item())


def pearson_corr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().float()
    targets = targets.detach().float()

    vp = preds - preds.mean()
    vt = targets - targets.mean()
    pcc = (vp * vt).mean() / (vp.pow(2).mean().sqrt() * vt.pow(2).mean().sqrt() + 1e-8)
    return float(pcc.item())


def mean_absolute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().float()
    targets = targets.detach().float()
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
# 13. Multi-Clip Training Loop
# ============================================
def run_one_epoch_multiclip(
    backbone,
    head,
    data_loader,
    optimizer=None,
    scaler=None,
    train=True,
    accum_steps: int = 1,
    use_amp: bool = True,
    bin_threshold: float = 0.0,
    grad_clip_norm: float = None,  # 🔧 新增參數
):
    """
    Multi-Clip 版本的訓練/驗證 loop

    輸入: [B, K, C, T, H, W]
    處理: Flatten 成 [B*K, C, T, H, W] 一起 forward
    聚合: 將 [B*K] 預測值 reshape 成 [B, K]，再 mean 成 [B]

    Args:
        grad_clip_norm: Gradient clipping 的 max norm（None = 不使用）
    """
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
    log_interval = max(1, total_steps // 20)
    epoch_start_time = time.time()
    batch_start = time.time()

    # 用於追蹤梯度統計（debug 用）
    grad_norms = []

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
                outputs = backbone(pixel_values=pixel_values_flat)

                if isinstance(outputs, torch.Tensor):
                    encoder_features = outputs
                elif hasattr(outputs, 'last_hidden_state'):
                    encoder_features = outputs.last_hidden_state
                else:
                    encoder_features = outputs[0] if isinstance(outputs, tuple) else outputs

                # Mean pooling if needed
                if encoder_features.dim() == 3:
                    feats = encoder_features.mean(dim=1)
                else:
                    feats = encoder_features

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
                    # ✅ Gradient Clipping（正確的做法）
                    if scaler is not None:
                        # 1. Unscale gradients before clipping
                        scaler.unscale_(optimizer)

                    # 2. Clip gradients
                    if grad_clip_norm is not None:
                        # 收集所有需要 clip 的參數
                        params_to_clip = []
                        params_to_clip.extend([p for p in backbone.parameters() if p.requires_grad])
                        params_to_clip.extend([p for p in head.parameters() if p.requires_grad])

                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)
                        grad_norms.append(grad_norm.item())

                    # 3. Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

        preds_f32 = preds.float().detach()
        labels_f32 = labels.float().detach()

        losses.append(raw_loss.item())
        all_preds.append(preds_f32.cpu())
        all_targets.append(labels_f32.cpu())

        batch_time = time.time() - batch_start

        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - epoch_start_time
            steps_done = step + 1
            steps_remain = total_steps - steps_done
            avg_batch_time = elapsed / steps_done
            eta_seconds = avg_batch_time * steps_remain
            eta_min = eta_seconds / 60
            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB"

            # 顯示平均 gradient norm
            grad_info = ""
            if grad_norms and grad_clip_norm is not None:
                avg_grad_norm = np.mean(grad_norms[-10:])  # 最近 10 步的平均
                grad_info = f" | grad: {avg_grad_norm:.2f}"

            print(f"  [{steps_done:4d}/{total_steps}] | "
                  f"loss={raw_loss.item():.4f} | "
                  f"batch={batch_time:.1f}s | "
                  f"clips={B*K}{grad_info} | "
                  f"ETA: {eta_min:.1f} min{gpu_mem}")

        batch_start = time.time()

    if train and optimizer is not None and step_count % accum_steps != 0:
        # 處理最後不足一個 accumulation cycle 的 gradient
        if scaler is not None:
            scaler.unscale_(optimizer)

        if grad_clip_norm is not None:
            params_to_clip = []
            params_to_clip.extend([p for p in backbone.parameters() if p.requires_grad])
            params_to_clip.extend([p for p in head.parameters() if p.requires_grad])
            torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    avg_ccc = concordance_cc(all_preds, all_targets)
    avg_pcc = pearson_corr(all_preds, all_targets)
    avg_mae = mean_absolute_error(all_preds, all_targets)
    avg_acc, avg_f1 = binary_metrics_from_valence(all_preds, all_targets, threshold=bin_threshold)

    # 返回梯度統計（用於監控）
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

    return avg_loss, avg_ccc, avg_pcc, avg_mae, avg_acc, avg_f1, avg_grad_norm


def collect_preds_targets_multiclip(backbone, head, data_loader, use_amp: bool = True):
    """Multi-Clip 版本的預測收集"""
    backbone.eval()
    head.eval()
    all_preds, all_targets = [], []

    total_steps = len(data_loader)
    log_interval = max(1, total_steps // 10)
    start_time = time.time()

    with torch.no_grad():
        for step, (pixel_values, labels) in enumerate(data_loader):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            B, K = pixel_values.shape[:2]
            pixel_values_flat = pixel_values.view(B * K, *pixel_values.shape[2:])

            with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = backbone(pixel_values=pixel_values_flat)

                if isinstance(outputs, torch.Tensor):
                    encoder_features = outputs
                elif hasattr(outputs, 'last_hidden_state'):
                    encoder_features = outputs.last_hidden_state
                else:
                    encoder_features = outputs[0] if isinstance(outputs, tuple) else outputs

                if encoder_features.dim() == 3:
                    feats = encoder_features.mean(dim=1)
                else:
                    feats = encoder_features

                preds_flat = head(feats)
                preds_per_clip = preds_flat.view(B, K)
                preds = preds_per_clip.mean(dim=1)

            all_preds.append(preds.float().cpu())
            all_targets.append(labels.float().cpu())

            if (step + 1) % log_interval == 0 or step == total_steps - 1:
                elapsed = time.time() - start_time
                steps_done = step + 1
                eta_seconds = (elapsed / steps_done) * (total_steps - steps_done)
                print(f"  Collecting [{steps_done}/{total_steps}] | ETA: {eta_seconds/60:.1f} min")

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets


# ============================================
# 14. Phase 1：只訓練 head（backbone 全部 freeze）
# ============================================
if SKIP_PHASE1_IF_EXISTS and os.path.exists(CKPT_P1_BEST):
    print("\n" + "=" * 60)
    print("✅ 偵測到 Phase 1 best checkpoint 已存在，跳過 Phase 1")
    print(f"   Checkpoint: {CKPT_P1_BEST}")
    ckpt = torch.load(CKPT_P1_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    best_val_ccc_p1 = ckpt.get("best_val_ccc", -1e9)
    print(f"   Phase 1 best val CCC: {best_val_ccc_p1:.4f}")
    print("=" * 60)
    history_p1 = []
else:
    print("\n" + "=" * 60)
    print("Phase 1: 只訓練 head（backbone 全部凍結）")
    print("=" * 60)
    print(f"  Batch Size: {BATCH_SIZE_P1}")
    print(f"  LR: {LR_HEAD_P1}, Weight Decay: {WEIGHT_DECAY}")
    print(f"  Dropout: {DROPOUT_HEAD}")

    for p in backbone.parameters():
        p.requires_grad = False

    optimizer_p1 = torch.optim.AdamW(head.parameters(), lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)
    scaler_p1 = GradScaler() if USE_AMP else None

    start_epoch_p1 = 0
    best_val_ccc_p1 = -1e9
    epochs_no_improve = 0

    if RESUME_PHASE1 and os.path.exists(CKPT_P1_LAST):
        print(f"🔄 Phase 1: 從 {CKPT_P1_LAST} 恢復...")
        ckpt = torch.load(CKPT_P1_LAST, map_location=device)
        backbone.load_state_dict(ckpt["backbone_state"])
        head.load_state_dict(ckpt["head_state"])
        optimizer_p1.load_state_dict(ckpt["optim_state"])
        if "scaler_state" in ckpt and scaler_p1 is not None and ckpt["scaler_state"] is not None:
            scaler_p1.load_state_dict(ckpt["scaler_state"])
        start_epoch_p1 = ckpt.get("epoch", 0) + 1
        best_val_ccc_p1 = ckpt.get("best_val_ccc", -1e9)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"Phase 1 從 epoch {start_epoch_p1} 繼續，best_val_ccc = {best_val_ccc_p1:.4f}")

    history_p1 = []

    for epoch in range(start_epoch_p1, MAX_EPOCHS_P1):
        epoch_start = time.time()
        print("\n" + "-" * 60)
        print(f"[Phase 1] Epoch {epoch+1}/{MAX_EPOCHS_P1}")
        print("-" * 60)

        print("Training...")
        train_loss, train_ccc, train_pcc, train_mae, train_acc, train_f1, train_grad_norm = run_one_epoch_multiclip(
            backbone, head, train_loader_p1,
            optimizer=optimizer_p1,
            scaler=scaler_p1,
            train=True,
            accum_steps=ACCUM_STEPS_P1,
            use_amp=USE_AMP,
            bin_threshold=0.0,
            grad_clip_norm=GRAD_CLIP_NORM,
        )

        print("Validating...")
        val_loss, val_ccc, val_pcc, val_mae, val_acc, val_f1, _ = run_one_epoch_multiclip(
            backbone, head, valid_loader_p1,
            optimizer=None,
            scaler=None,
            train=False,
            accum_steps=1,
            use_amp=USE_AMP,
            bin_threshold=0.0,
            grad_clip_norm=None,
        )

        epoch_time = time.time() - epoch_start

        if epoch == start_epoch_p1:
            print(f"\n📊 人臉偵測統計：")
            print(f"  Train: {train_dataset.get_face_stats()}")
            print(f"  Valid: {valid_dataset.get_face_stats()}")

        overfit_gap = train_ccc - val_ccc

        print(f"\n{'='*60}")
        print(f"[Train] loss={train_loss:.4f}, CCC={train_ccc:.4f}, PCC={train_pcc:.4f}, "
              f"MAE={train_mae:.4f}, acc={train_acc:.4f}, F1={train_f1:.4f}")
        if train_grad_norm > 0:
            print(f"        grad_norm={train_grad_norm:.3f}")
        print(f"[Valid] loss={val_loss:.4f}, CCC={val_ccc:.4f}, PCC={val_pcc:.4f}, "
              f"MAE={val_mae:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}")
        print(f"[Gap] Train-Valid CCC: {overfit_gap:.4f}")
        print(f"Epoch time: {epoch_time/60:.1f} min")
        print(f"{'='*60}")

        ckpt_state = {
            "phase": 1,
            "experiment_name": PHASE1_EXPERIMENT_NAME,
            "epoch": epoch,
            "backbone_state": backbone.state_dict(),
            "head_state": head.state_dict(),
            "optim_state": optimizer_p1.state_dict(),
            "scaler_state": scaler_p1.state_dict() if scaler_p1 is not None else None,
            "best_val_ccc": best_val_ccc_p1,
            "epochs_no_improve": epochs_no_improve,
        }
        torch.save(ckpt_state, CKPT_P1_LAST)

        history_p1.append({
            "phase": 1,
            "experiment_name": PHASE1_EXPERIMENT_NAME,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ccc": train_ccc,
            "train_pcc": train_pcc,
            "train_mae": train_mae,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_grad_norm": train_grad_norm,
            "val_loss": val_loss,
            "val_ccc": val_ccc,
            "val_pcc": val_pcc,
            "val_mae": val_mae,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "overfit_gap": overfit_gap,
        })

        history_df = pd.DataFrame(history_p1)
        history_df.to_csv(os.path.join(PHASE1_CKPT_DIR, "training_history_p1.csv"), index=False)

        if val_ccc > best_val_ccc_p1:
            best_val_ccc_p1 = val_ccc
            epochs_no_improve = 0
            ckpt_state["best_val_ccc"] = best_val_ccc_p1
            torch.save(ckpt_state, CKPT_P1_BEST)
            print(f"✅ Phase 1: val CCC 提升，儲存 best checkpoint：{best_val_ccc_p1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Phase 1: val CCC 無改善，連續 {epochs_no_improve}/{PATIENCE_P1} epoch")

        if epochs_no_improve >= PATIENCE_P1:
            print("⏹ Phase 1: 觸發 early stopping")
            break

    if train_dataset.skipped_videos:
        print(f"\n⚠️ Phase 1 訓練期間跳過了 {len(train_dataset.skipped_videos)} 個有問題的影片")

    print(f"\n✅ Phase 1 結束，最佳 val CCC = {best_val_ccc_p1:.4f}")


# ============================================
# 15. Phase 2：Multi-Clip Fine-tuning
# ============================================
print("\n" + "=" * 70)
print(f"Phase 2: {PHASE2_EXPERIMENT_NAME}")
print(f"Multi-Clip Training (K={NUM_CLIPS})")
print(f"backbone 最後 {N_UNFREEZE_LAYERS} 層 + head 一起 fine-tune")
print("=" * 70)

print(f"\n📋 Phase 2 實驗設定:")
print(f"  Num Clips: {NUM_CLIPS}")
print(f"  Batch Size: {BATCH_SIZE_P2} (effective: {BATCH_SIZE_P2 * ACCUM_STEPS_P2})")
print(f"  每次 Forward: {BATCH_SIZE_P2 * NUM_CLIPS} clips")
print(f"  LR Head: {LR_HEAD_P2}, LR Backbone: {LR_BACKBONE_P2}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Unfreeze Layers: {N_UNFREEZE_LAYERS}")
print(f"  Mixup: {USE_MIXUP_P2}")
print(f"  Huber Delta: {HUBER_DELTA}, CCC Weight: {CCC_WEIGHT}")
print(f"  Phase 1 來源: {CKPT_P1_BEST}")

for var_name in ['optimizer_p1', 'scaler_p1']:
    if var_name in dir():
        exec(f'del {var_name}')

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f"\nGPU Memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

if os.path.exists(CKPT_P1_BEST):
    ckpt = torch.load(CKPT_P1_BEST, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    print(f"✅ 已載入 Phase 1 best checkpoint: {CKPT_P1_BEST}")
    print(f"   Phase 1 val CCC: {ckpt.get('best_val_ccc', 'N/A')}")
else:
    print("❌ 找不到 Phase 1 best checkpoint！請先執行 Phase 1 訓練。")
    print(f"   預期路徑: {CKPT_P1_BEST}")
    raise FileNotFoundError(f"Phase 1 checkpoint not found: {CKPT_P1_BEST}")

for p in backbone.parameters():
    p.requires_grad = False

def unfreeze_last_n_layers(backbone, n_unfreeze: int = 4):
    print(f"\n🔍 檢查 backbone 結構...")

    all_param_names = [n for n, p in backbone.named_parameters()]
    print(f"總共 {len(all_param_names)} 個參數")

    layer_indices = set()
    layer_pattern = re.compile(r'blocks\.(\d+)\.')
    alt_pattern = re.compile(r'layer\.(\d+)\.')
    alt_pattern2 = re.compile(r'layers\.(\d+)\.')

    for name in all_param_names:
        for pattern in [layer_pattern, alt_pattern, alt_pattern2]:
            match = pattern.search(name)
            if match:
                layer_indices.add(int(match.group(1)))

    if layer_indices:
        print(f"\n找到的 layer 索引: {sorted(layer_indices)}")
        total_layers = max(layer_indices) + 1
    else:
        total_layers = getattr(backbone.config, "num_hidden_layers", 12)

    print(f"Total encoder layers: {total_layers}")

    target_indices = list(range(max(0, total_layers - n_unfreeze), total_layers))
    print(f"目標解凍層: {target_indices}")

    unfrozen_params = []

    for name, param in backbone.named_parameters():
        for idx in target_indices:
            patterns = [
                f"blocks.{idx}.",
                f"encoder.blocks.{idx}.",
                f"encoder.layer.{idx}.",
                f"encoder.layers.{idx}.",
                f"videomae.encoder.layer.{idx}.",
                f"layer.{idx}.",
                f"layers.{idx}.",
            ]
            if any(pat in name for pat in patterns):
                param.requires_grad = True
                unfrozen_params.append(name)
                break

    if unfrozen_params:
        print(f"✅ 解凍了 {len(unfrozen_params)} 個參數張量")
        print(f"   解凍的層: {target_indices}")
    else:
        print("⚠️ 標準模式找不到，嘗試根據實際參數名稱解凍...")
        for name, param in backbone.named_parameters():
            for pattern in [layer_pattern, alt_pattern, alt_pattern2]:
                match = pattern.search(name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx in target_indices:
                        param.requires_grad = True
                        unfrozen_params.append(name)
                    break

        if unfrozen_params:
            print(f"✅ 使用正則表達式解凍了 {len(unfrozen_params)} 個參數")
        else:
            print("❌ 無法解凍任何 backbone 參數！")

    return len(unfrozen_params)

num_unfrozen = unfreeze_last_n_layers(backbone, n_unfreeze=N_UNFREEZE_LAYERS)

for p in head.parameters():
    p.requires_grad = True

backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
total_trainable = backbone_trainable + head_trainable
print(f"\nTrainable params - backbone: {backbone_trainable:,}, head: {head_trainable:,}")
print(f"Total trainable: {total_trainable:,}")

backbone_trainable_params = [p for p in backbone.parameters() if p.requires_grad]

optimizer_p2 = torch.optim.AdamW(
    [
        {"params": head.parameters(), "lr": LR_HEAD_P2},
        {"params": backbone_trainable_params, "lr": LR_BACKBONE_P2},
    ],
    weight_decay=WEIGHT_DECAY
)

scaler_p2 = GradScaler() if USE_AMP else None

print(f"\n🔄 建立 Phase 2 DataLoader (batch_size={BATCH_SIZE_P2})...")

# Phase 2 使用新的 generator
g_p2 = torch.Generator()
g_p2.manual_seed(42)

train_loader_p2 = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_P2,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_multi_clips,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
    generator=g_p2,
)

valid_loader_p2 = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE_P2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_multi_clips,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

print(f"Train batches: {len(train_loader_p2)}, Valid batches: {len(valid_loader_p2)}")

start_epoch_p2 = 0
best_val_ccc_p2 = -1e9
epochs_no_improve = 0

if RESUME_PHASE2 and os.path.exists(CKPT_P2_LAST):
    print(f"🔄 Phase 2: 從 {CKPT_P2_LAST} 恢復...")
    ckpt2 = torch.load(CKPT_P2_LAST, map_location=device)
    backbone.load_state_dict(ckpt2["backbone_state"])
    head.load_state_dict(ckpt2["head_state"])
    optimizer_p2.load_state_dict(ckpt2["optim_state"])
    if "scaler_state" in ckpt2 and scaler_p2 is not None and ckpt2["scaler_state"] is not None:
        scaler_p2.load_state_dict(ckpt2["scaler_state"])
    start_epoch_p2 = ckpt2.get("epoch", 0) + 1
    best_val_ccc_p2 = ckpt2.get("best_val_ccc", -1e9)
    epochs_no_improve = ckpt2.get("epochs_no_improve", 0)
    print(f"Phase 2 從 epoch {start_epoch_p2} 繼續，best_val_ccc = {best_val_ccc_p2:.4f}")

train_dataset.skipped_videos = []

history_p2 = []

for epoch in range(start_epoch_p2, MAX_EPOCHS_P2):
    epoch_start = time.time()
    print("\n" + "-" * 70)
    print(f"[Phase 2 - {PHASE2_EXPERIMENT_NAME}] Epoch {epoch+1}/{MAX_EPOCHS_P2}")
    print("-" * 70)

    print("Training...")
    train_loss, train_ccc, train_pcc, train_mae, train_acc, train_f1, train_grad_norm = run_one_epoch_multiclip(
        backbone, head, train_loader_p2,
        optimizer=optimizer_p2,
        scaler=scaler_p2,
        train=True,
        accum_steps=ACCUM_STEPS_P2,
        use_amp=USE_AMP,
        bin_threshold=0.0,
        grad_clip_norm=GRAD_CLIP_NORM,
    )

    print("Validating...")
    val_loss, val_ccc, val_pcc, val_mae, val_acc, val_f1, _ = run_one_epoch_multiclip(
        backbone, head, valid_loader_p2,
        optimizer=None,
        scaler=None,
        train=False,
        accum_steps=1,
        use_amp=USE_AMP,
        bin_threshold=0.0,
        grad_clip_norm=None,
    )

    epoch_time = time.time() - epoch_start
    overfit_gap = train_ccc - val_ccc

    print(f"\n{'='*70}")
    print(f"[Train] loss={train_loss:.4f}, CCC={train_ccc:.4f}, PCC={train_pcc:.4f}, "
          f"MAE={train_mae:.4f}, acc={train_acc:.4f}, F1={train_f1:.4f}")
    if train_grad_norm > 0:
        print(f"        grad_norm={train_grad_norm:.3f}")
    print(f"[Valid] loss={val_loss:.4f}, CCC={val_ccc:.4f}, PCC={val_pcc:.4f}, "
          f"MAE={val_mae:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}")
    print(f"[Gap] Train-Valid CCC: {overfit_gap:.4f}")
    print(f"Epoch time: {epoch_time/60:.1f} min")
    print(f"{'='*70}")

    ckpt_state2 = {
        "phase": 2,
        "experiment_name": PHASE2_EXPERIMENT_NAME,
        "phase1_source": PHASE1_EXPERIMENT_NAME,
        "epoch": epoch,
        "backbone_state": backbone.state_dict(),
        "head_state": head.state_dict(),
        "optim_state": optimizer_p2.state_dict(),
        "scaler_state": scaler_p2.state_dict() if scaler_p2 is not None else None,
        "best_val_ccc": best_val_ccc_p2,
        "epochs_no_improve": epochs_no_improve,
    }
    torch.save(ckpt_state2, CKPT_P2_LAST)

    history_p2.append({
        "phase": 2,
        "experiment_name": PHASE2_EXPERIMENT_NAME,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_ccc": train_ccc,
        "train_pcc": train_pcc,
        "train_mae": train_mae,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "train_grad_norm": train_grad_norm,
        "val_loss": val_loss,
        "val_ccc": val_ccc,
        "val_pcc": val_pcc,
        "val_mae": val_mae,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "overfit_gap": overfit_gap,
    })

    history_p2_df = pd.DataFrame(history_p2)
    history_p2_df.to_csv(os.path.join(PHASE2_CKPT_DIR, "training_history_p2.csv"), index=False)

    if val_ccc > best_val_ccc_p2:
        best_val_ccc_p2 = val_ccc
        epochs_no_improve = 0
        ckpt_state2["best_val_ccc"] = best_val_ccc_p2
        torch.save(ckpt_state2, CKPT_P2_BEST)
        print(f"✅ Phase 2: val CCC 提升，儲存 best checkpoint：{best_val_ccc_p2:.4f}")
    else:
        epochs_no_improve += 1
        print(f"⚠️ Phase 2: val CCC 無改善，連續 {epochs_no_improve}/{PATIENCE_P2} epoch")

    if epochs_no_improve >= PATIENCE_P2:
        print("⏹ Phase 2: 觸發 early stopping")
        break

if train_dataset.skipped_videos:
    print(f"\n⚠️ Phase 2 訓練期間跳過了 {len(train_dataset.skipped_videos)} 個有問題的影片")

print(f"\n✅ Phase 2 結束，最佳 val CCC = {best_val_ccc_p2:.4f}")

# ============================================
# 16. 在驗證集上搜尋最佳 binary threshold
# ============================================
print("\n" + "=" * 60)
print("驗證集上搜尋最佳二極性 threshold（以 F1 為主）")
print("=" * 60)

if os.path.exists(CKPT_P2_BEST):
    best_ckpt2 = torch.load(CKPT_P2_BEST, map_location=device)
    backbone.load_state_dict(best_ckpt2["backbone_state"])
    head.load_state_dict(best_ckpt2["head_state"])
    print("✅ 已載入 Phase 2 最佳 checkpoint 進行 threshold 搜尋")
else:
    print("⚠️ 找不到 Phase 2 best checkpoint，使用目前模型")

print("\n收集驗證集預測...")
val_preds_all, val_targets_all = collect_preds_targets_multiclip(backbone, head, valid_loader_p2, use_amp=USE_AMP)

thresholds = np.linspace(-0.5, 0.5, 101)
best_thr = 0.0
best_f1 = -1.0
best_acc = -1.0

for thr in thresholds:
    acc_thr, f1_thr = binary_metrics_from_valence(val_preds_all, val_targets_all, threshold=float(thr))
    if (f1_thr > best_f1) or (math.isclose(f1_thr, best_f1) and acc_thr > best_acc):
        best_f1 = f1_thr
        best_acc = acc_thr
        best_thr = float(thr)

print(f"\n最佳 threshold（valid）: {best_thr:.4f}")
print(f"  Valid F1: {best_f1:.4f}")
print(f"  Valid Acc: {best_acc:.4f}")

with open(BIN_THRESH_PATH, "w", encoding="utf-8") as f:
    f.write(f"best_threshold={best_thr:.6f}\n")
    f.write(f"valid_f1={best_f1:.6f}\n")
    f.write(f"valid_acc={best_acc:.6f}\n")

print(f"最佳 threshold 已儲存至：{BIN_THRESH_PATH}")

# ============================================
# 17. 測試集評估
# ============================================
print("\n" + "=" * 60)
print("測試集評估")
print("=" * 60)

if os.path.exists(CKPT_P2_BEST):
    print("載入 Phase 2 最佳 checkpoint 進行測試")
    best_ckpt2 = torch.load(CKPT_P2_BEST, map_location=device)
    backbone.load_state_dict(best_ckpt2["backbone_state"])
    head.load_state_dict(best_ckpt2["head_state"])
else:
    print("⚠️ 找不到 Phase 2 best checkpoint，使用目前模型")

# ✅ 在測試階段才建立 test_loader（節省記憶體和時間）
if test_loader is None:
    print(f"\n🔄 建立 Test DataLoader (batch_size={BATCH_SIZE_P2})...")
    TEST_BATCH_SIZE = BATCH_SIZE_P2
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_multi_clips,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    print(f"Test batches: {len(test_loader)}")
    print(f"每次 forward: {TEST_BATCH_SIZE} × {NUM_CLIPS} = {TEST_BATCH_SIZE * NUM_CLIPS} clips")

print("\n評估測試集 (threshold=0.0)...")
test_loss, test_ccc, test_pcc, test_mae, test_acc_zero, test_f1_zero, _ = run_one_epoch_multiclip(
    backbone, head, test_loader,
    optimizer=None,
    scaler=None,
    train=False,
    accum_steps=1,
    use_amp=USE_AMP,
    bin_threshold=0.0,
    grad_clip_norm=None,
)

print("\n收集測試集預測（用於最佳 threshold 評估）...")
test_preds_all, test_targets_all = collect_preds_targets_multiclip(backbone, head, test_loader, use_amp=USE_AMP)
test_acc_best, test_f1_best = binary_metrics_from_valence(test_preds_all, test_targets_all, threshold=best_thr)

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

# ============================================
# 18. 儲存訓練歷史與問題影片清單
# ============================================
all_skipped = set(train_dataset.skipped_videos + valid_dataset.skipped_videos + test_dataset.skipped_videos)
if all_skipped:
    skipped_path = os.path.join(PHASE2_CKPT_DIR, "skipped_videos.txt")
    with open(skipped_path, "w", encoding="utf-8") as f:
        for v in sorted(all_skipped):
            f.write(v + "\n")
    print(f"被跳過的影片清單已儲存至：{skipped_path}")
    print(f"共 {len(all_skipped)} 個影片有問題")

summary_path = os.path.join(PHASE2_CKPT_DIR, "final_results.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(f"VideoMAE V2 Valence Regression (Multi-Clip)\n")
    f.write(f"Phase 2 Experiment: {PHASE2_EXPERIMENT_NAME}\n")
    f.write("=" * 70 + "\n\n")
    f.write("=== Phase 1 來源 ===\n")
    f.write(f"實驗名稱: {PHASE1_EXPERIMENT_NAME}\n")
    f.write(f"Checkpoint: {CKPT_P1_BEST}\n")
    f.write(f"Phase 1 best val CCC: {best_val_ccc_p1:.4f}\n\n")
    f.write("=== Phase 2 設定 ===\n")
    for k, v in phase2_config.items():
        f.write(f"{k}: {v}\n")
    f.write("\n=== 結果 ===\n")
    f.write(f"Phase 2 best val CCC: {best_val_ccc_p2:.4f}\n\n")
    f.write(f"Best binary threshold: {best_thr:.4f}\n\n")
    f.write("[Test Results - threshold=0.0]\n")
    f.write(f"  Loss: {test_loss:.4f}\n")
    f.write(f"  CCC:  {test_ccc:.4f}\n")
    f.write(f"  PCC:  {test_pcc:.4f}\n")
    f.write(f"  MAE:  {test_mae:.4f}\n")
    f.write(f"  Acc:  {test_acc_zero:.4f}\n")
    f.write(f"  F1:   {test_f1_zero:.4f}\n\n")
    f.write(f"[Test Results - best threshold={best_thr:.4f}]\n")
    f.write(f"  Acc:  {test_acc_best:.4f}\n")
    f.write(f"  F1:   {test_f1_best:.4f}\n\n")
    if all_skipped:
        f.write(f"Skipped videos: {len(all_skipped)}\n")

print(f"最終結果摘要已儲存至：{summary_path}")

print("\n🎉 Multi-Clip Training 完成！")
print(f"\n📊 Multi-Clip 設定摘要：")
print(f"  Num Clips: {NUM_CLIPS}")
print(f"  Sampling: Train=Random, Valid/Test=Center")
print(f"  Aggregation: Mean across clips")
print(f"  Batch Size: {BATCH_SIZE_P2} (effective: {BATCH_SIZE_P2 * ACCUM_STEPS_P2})")
print(f"  Forward Clips per Batch: {BATCH_SIZE_P2 * NUM_CLIPS}")
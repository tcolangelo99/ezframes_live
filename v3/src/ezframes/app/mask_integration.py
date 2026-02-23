from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_INDEX_BASE_CACHE: dict[str, int] = {}


def video_workspace_dir(workspace_root: str, video_path: str) -> Path:
    stem = Path(video_path).stem
    return Path(workspace_root) / stem


def mask_dir_for_video(workspace_root: str, video_path: str) -> Path:
    return video_workspace_dir(workspace_root, video_path) / "binary_masks"


def _detect_mask_index_base(mask_dir: Path) -> int:
    key = str(mask_dir.resolve()) if mask_dir.exists() else str(mask_dir)
    cached = _INDEX_BASE_CACHE.get(key)
    if cached is not None:
        return cached

    min_index: int | None = None
    for mask_path in mask_dir.glob("*.png"):
        stem = mask_path.stem
        if not stem.isdigit():
            continue
        value = int(stem)
        if min_index is None or value < min_index:
            min_index = value

    # Default to 0-based when unknown, which matches Cutie export behavior.
    base = 0 if min_index in (None, 0) else 1
    _INDEX_BASE_CACHE[key] = base
    return base


def load_mask(mask_dir: Path, frame_number: int) -> np.ndarray | None:
    index_base = _detect_mask_index_base(mask_dir)

    # EzFrames frame counter is 1-based in processing loop.
    # Cutie exports can be 0-based or 1-based depending on workflow/build.
    if index_base == 0:
        target = max(0, int(frame_number) - 1)
    else:
        target = int(frame_number)

    candidates = [mask_dir / f"{target:07d}.png"]
    if index_base == 0:
        # Safe fallback for mixed/legacy folders.
        candidates.append(mask_dir / f"{int(frame_number):07d}.png")
    else:
        candidates.append(mask_dir / f"{max(0, int(frame_number) - 1):07d}.png")

    for mask_path in candidates:
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        return (mask > 0).astype(np.uint8) * 255

    return None

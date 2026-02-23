from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ezframes.common.config import AppPaths


log = logging.getLogger(__name__)


def _is_usable_ffmpeg(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = subprocess.run(
            [str(path), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def resolve_ffmpeg(paths: AppPaths) -> str:
    env_ffmpeg = os.environ.get("EZFRAMES_FFMPEG_PATH", "").strip()
    candidates: list[Path] = []
    if env_ffmpeg:
        candidates.append(Path(env_ffmpeg))

    for root in paths.source_roots():
        candidates.extend(
            [
                root / "ffmpeg" / "bin" / "ffmpeg.exe",
                root / "ffmpeg" / "bin" / "ffmpeg",
                root / "assets" / "ffmpeg" / "bin" / "ffmpeg.exe",
                root / "assets" / "ffmpeg" / "bin" / "ffmpeg",
            ]
        )
    candidates.extend(
        [
            paths.assets_dir / "ffmpeg" / "bin" / "ffmpeg.exe",
            paths.install_root / "assets" / "ffmpeg" / "bin" / "ffmpeg.exe",
        ]
    )

    for c in candidates:
        if _is_usable_ffmpeg(c):
            return str(c)
        if c.exists():
            log.warning("Skipping unusable ffmpeg candidate: %s", c)

    found = shutil.which("ffmpeg")
    if found and _is_usable_ffmpeg(Path(found)):
        return found

    raise FileNotFoundError(
        "ffmpeg executable was not found. Set EZFRAMES_FFMPEG_PATH or place ffmpeg under assets/ffmpeg/bin."
    )


class FFmpegWriter:
    def __init__(
        self,
        ffmpeg_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        pix_fmt: str = "bgr24",
        profile: str = "standard",
    ):
        self.output_path = str(Path(output_path).with_suffix(".mov"))
        self.command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            pix_fmt,
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "prores_ks",
            "-profile:v",
            profile,
            "-pix_fmt",
            "yuv422p10le" if profile not in {"4444", "4444xq"} else "yuva444p10le",
            self.output_path,
        ]
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "FFmpegWriter":
        log.info("Starting ffmpeg: %s", " ".join(self.command))
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("FFmpeg process not started")
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError as exc:
            rc = self.proc.poll()
            details = f"returncode={rc}" if rc is not None else "process still running"
            if rc is not None and self.proc.stderr is not None:
                try:
                    stderr_text = self.proc.stderr.read().decode(errors="ignore").strip()
                    if stderr_text:
                        details = f"{details}; stderr={stderr_text}"
                except Exception:
                    pass
            raise RuntimeError(f"FFmpeg stdin closed unexpectedly while encoding ({details}).") from exc

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        if exc_type is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            return

        try:
            stdout, stderr = self.proc.communicate(timeout=120)
        except subprocess.TimeoutExpired as timeout_exc:
            self.proc.kill()
            stdout, stderr = self.proc.communicate()
            raise RuntimeError(
                f"FFmpeg timed out while finishing encode:\n{stderr.decode(errors='ignore')}\n{stdout.decode(errors='ignore')}"
            ) from timeout_exc

        if self.proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode(errors='ignore')}\n{stdout.decode(errors='ignore')}")


def open_capture(video_path: str) -> tuple[cv2.VideoCapture, int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, width, height, fps, total

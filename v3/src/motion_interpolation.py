from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch


class MotionInterpolator:
    def __init__(self):
        self.model = None
        self.device_name = self._resolve_device_name()
        self.device = torch.device(self.device_name)
        print(f"Using device: {self.device_name}")

    @staticmethod
    def _is_cuda_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "cuda",
            "cudnn",
            "nvrtc",
            "device-side assert",
            "out of memory",
            "driver",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _resolve_device_name() -> str:
        requested = os.environ.get("EZFRAMES_RIFE_DEVICE", "auto").strip().lower()
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _set_device(self, device_name: str) -> None:
        device_name = device_name.strip().lower()
        if device_name not in {"cpu", "cuda"}:
            device_name = "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device_name = device_name
        self.device = torch.device(self.device_name)
        os.environ["EZFRAMES_RIFE_DEVICE"] = self.device_name
        print(f"RIFE device set to: {self.device_name}")

    @staticmethod
    def _resolve_model_path() -> Path:
        env_path = os.environ.get("EZFRAMES_FLOWNET_PATH", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        base = Path(__file__).resolve().parent
        candidates = [
            base / "flownet.pkl",
            base / "models" / "flownet.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("flownet.pkl")

    @staticmethod
    def _import_model_class():
        import RIFE_HDv3

        importlib.reload(RIFE_HDv3)
        return RIFE_HDv3.Model

    def _load_model_on_current_device(self, model_path: Path) -> None:
        os.environ["EZFRAMES_RIFE_DEVICE"] = self.device_name
        model_cls = self._import_model_class()
        self.model = model_cls()
        self.model.load_model(str(model_path), -1)
        self.model.eval()
        self.model.device()

    def load_model(self):
        print("Loading RIFE model...")
        model_path = self._resolve_model_path()
        try:
            self._load_model_on_current_device(model_path)
        except Exception as exc:
            if self.device.type == "cuda" and self._is_cuda_error(exc):
                print(f"RIFE CUDA load failed, falling back to CPU: {exc}")
                self._set_device("cpu")
                self._load_model_on_current_device(model_path)
            else:
                raise
        print("RIFE model loaded successfully.")

    def load_image(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1)).float()[None].to(
            self.device, non_blocking=(self.device.type == "cuda")
        ) / 255.0

    def _opencv_cuda_available(self) -> bool:
        if self.device.type != "cuda":
            return False
        try:
            return bool(hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0)
        except Exception:
            return False

    def resize_frame_cuda(self, frame, width, height):
        if self._opencv_cuda_available():
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (width, height))
                return gpu_resized.download()
            except Exception:
                pass
        return cv2.resize(frame, (width, height))

    def process_frame(self, frame, width, height, expected_frame_size):
        frame_resized = self.resize_frame_cuda(frame, width, height)
        actual_frame_size = frame_resized.nbytes
        assert actual_frame_size == expected_frame_size, f"Expected {expected_frame_size} bytes, got {actual_frame_size}"
        return frame_resized

    @staticmethod
    def _resolve_ffmpeg_path() -> str:
        ffmpeg_path = os.environ.get("EZFRAMES_FFMPEG_PATH", "").strip()
        if ffmpeg_path and Path(ffmpeg_path).exists():
            return ffmpeg_path

        base = Path(__file__).resolve().parent
        candidates = [
            base / "ffmpeg" / "bin" / "ffmpeg.exe",
            base / "ffmpeg" / "bin" / "ffmpeg",
            base / "assets" / "ffmpeg" / "bin" / "ffmpeg.exe",
            base / "assets" / "ffmpeg" / "bin" / "ffmpeg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg

        raise RuntimeError("ffmpeg not found. Set EZFRAMES_FFMPEG_PATH or install ffmpeg.")

    def _interpolate_video_once(self, input_path, output_path, sf=2, prores_quality=3, progress_callback=None):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frame_size = width * height * 3

        output_path = os.path.splitext(output_path)[0] + ".mov"

        ffmpeg_path = self._resolve_ffmpeg_path()
        prores_profiles = ["proxy", "lt", "standard", "hq", "4444", "4444xq"]
        prores_profile = prores_profiles[max(1, min(prores_quality, len(prores_profiles))) - 1]
        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps * sf),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "prores_ks",
            "-profile:v",
            prores_profile,
            "-pix_fmt",
            "yuv422p10le" if prores_profile not in ["4444", "4444xq"] else "yuva444p10le",
            output_path,
        ]
        if os.name == "nt":
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            process.kill()
            raise RuntimeError("Failed to read the first frame")

        prev = self.load_image(prev_frame)
        frame_count = 0
        total_frames_to_write = max(1, total_frames * sf)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor, torch.no_grad():
                while True:
                    ret, curr_frame = cap.read()
                    if not ret:
                        break

                    curr = self.load_image(curr_frame)
                    frame = executor.submit(
                        self.process_frame,
                        (prev[0] * 255).byte().cpu().numpy().transpose(1, 2, 0),
                        width,
                        height,
                        expected_frame_size,
                    ).result()
                    if process.stdin is None:
                        raise RuntimeError("FFmpeg stdin is unavailable during interpolation.")
                    process.stdin.write(frame.tobytes())
                    frame_count += 1

                    for i in range(1, sf):
                        t = i / sf
                        middle = self.model.inference(prev, curr, t)
                        interpolated_frame = executor.submit(
                            self.process_frame,
                            (middle[0] * 255).byte().cpu().numpy().transpose(1, 2, 0),
                            width,
                            height,
                            expected_frame_size,
                        ).result()
                        process.stdin.write(interpolated_frame.tobytes())
                        frame_count += 1

                    if progress_callback:
                        progress_callback(frame_count, total_frames_to_write)

                    prev = curr

                last_frame = (prev[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                last_frame_resized = self.resize_frame_cuda(last_frame, width, height)
                if process.stdin is None:
                    raise RuntimeError("FFmpeg stdin is unavailable for final frame.")
                process.stdin.write(last_frame_resized.tobytes())
                frame_count += 1
        except Exception:
            process.kill()
            process.communicate()
            raise
        finally:
            cap.release()

        if process.stdin:
            process.stdin.close()
            process.stdin = None
        stdout, stderr = process.communicate(timeout=120)
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with error code {process.returncode}: {stderr.decode(errors='ignore')}")

        return frame_count, total_frames_to_write

    def interpolate_video(self, input_path, output_path, sf=2, prores_quality=3, progress_callback=None):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            return self._interpolate_video_once(
                input_path=input_path,
                output_path=output_path,
                sf=sf,
                prores_quality=prores_quality,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            if self.device.type == "cuda" and self._is_cuda_error(exc):
                print(f"RIFE CUDA interpolation failed, retrying on CPU: {exc}")
                self._set_device("cpu")
                self.load_model()
                final_output_path = os.path.splitext(output_path)[0] + ".mov"
                if os.path.exists(final_output_path):
                    try:
                        os.remove(final_output_path)
                    except Exception:
                        pass
                return self._interpolate_video_once(
                    input_path=input_path,
                    output_path=output_path,
                    sf=sf,
                    prores_quality=prores_quality,
                    progress_callback=progress_callback,
                )
            raise

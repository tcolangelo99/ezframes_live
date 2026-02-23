from __future__ import annotations

import logging
import math
import os
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from ezframes.app.auth_gate import enforce_authorized_launch
from ezframes.app.mask_integration import load_mask, mask_dir_for_video, video_workspace_dir
from ezframes.app.motion_detection import (
    MotionOptions,
    detect_camera_motion,
    detect_motion,
    draw_contour,
    visualize_homography,
)
from ezframes.app.rife_interpolation import InterpolationRequest, RifeInterpolator
from ezframes.app.video_io import FFmpegWriter, open_capture, resolve_ffmpeg
from ezframes.common.config import AppPaths
from ezframes.common.logging_utils import configure_logging


log = logging.getLogger(__name__)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class EzFramesUI:
    def __init__(self, root: tk.Tk, paths: AppPaths):
        self.root = root
        self.paths = paths
        self.last_output_path: str | None = None
        self.interpolator = RifeInterpolator(paths)

        self.preview_cap: cv2.VideoCapture | None = None
        self.preview_total_frames = 0
        self.preview_fps = 24.0
        self.preview_frame = 0
        self.preview_playing = False
        self.preview_job: str | None = None
        self.preview_next_tick_s: float | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self._cutie_python: str | None = None

        self._build_ui()
        self._warn_if_no_cuda()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("EzFrames v3")
        self.root.geometry("1240x830")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.var_video = tk.StringVar()
        self.var_output = tk.StringVar(value=str(self.paths.workspace_dir))
        self.var_status = tk.StringVar(value="Ready.")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_frame_label = tk.StringVar(value="No video loaded")
        self.var_min_area_label = tk.StringVar(value="Min Contour Area: 100 px")
        self.var_camera_label = tk.StringVar(value="Cam Motion Sensitivity: 25%")

        self.var_use_masks = tk.BooleanVar(value=True)
        self.var_output_masked = tk.BooleanVar(value=False)
        self.var_output_images = tk.BooleanVar(value=False)
        self.var_camera = tk.BooleanVar(value=False)
        self.var_debug = tk.BooleanVar(value=False)
        self.var_motion_method = tk.StringVar(value="Absdiff")
        self.var_detector = tk.StringVar(value="ORB")
        self.var_sift_scale = tk.DoubleVar(value=0.5)
        self.var_min_area_log = tk.DoubleVar(value=2.0)
        self.var_area_ratio = tk.DoubleVar(value=0.5)
        self.var_camera_threshold = tk.DoubleVar(value=0.75)
        self.var_prores = tk.StringVar(value="standard")
        self.var_cutie_objects = tk.IntVar(value=1)
        self.var_interp_factor = tk.IntVar(value=2)

        self._add_path_row(main, 0, "Video", self.var_video, self._browse_video)
        self._add_path_row(main, 1, "Output Dir", self.var_output, self._browse_output)

        opts = ttk.LabelFrame(main, text="Processing Options", padding=8)
        opts.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 6))

        self.use_masks_check = ttk.Checkbutton(opts, text="Use Cutie Masks", variable=self.var_use_masks, command=self._sync_mask_state)
        self.use_masks_check.grid(row=0, column=0, sticky="w", padx=3)
        self.output_masked_check = ttk.Checkbutton(opts, text="Output Masked Video (Alpha)", variable=self.var_output_masked)
        self.output_masked_check.grid(row=0, column=1, sticky="w", padx=3)
        ttk.Checkbutton(opts, text="Output Image Sequence", variable=self.var_output_images).grid(
            row=0, column=2, sticky="w", padx=3
        )
        ttk.Checkbutton(opts, text="Camera Motion Filter", variable=self.var_camera, command=self._sync_camera_state).grid(
            row=0, column=3, sticky="w", padx=3
        )
        ttk.Checkbutton(opts, text="Debug Output", variable=self.var_debug).grid(row=0, column=4, sticky="w", padx=3)

        ttk.Label(opts, text="Motion Method").grid(row=1, column=0, sticky="w")
        motion_combo = ttk.Combobox(opts, textvariable=self.var_motion_method, values=["Absdiff", "Optical Flow"], width=14, state="readonly")
        motion_combo.grid(row=1, column=1, sticky="w")
        motion_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_motion_method_change())
        ttk.Label(opts, text="Feature Detector").grid(row=1, column=2, sticky="w")
        self.detector_combo = ttk.Combobox(opts, textvariable=self.var_detector, values=["ORB", "SIFT"], width=10, state="readonly")
        self.detector_combo.grid(row=1, column=3, sticky="w")
        self.detector_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_sift_state())

        ttk.Label(opts, text="SIFT Downscale").grid(row=1, column=4, sticky="w")
        self.sift_slider = ttk.Scale(opts, from_=0.3, to=1.0, variable=self.var_sift_scale, orient=tk.HORIZONTAL, length=140)
        self.sift_slider.grid(row=1, column=5, sticky="w")

        ttk.Label(opts, textvariable=self.var_min_area_label).grid(row=2, column=0, sticky="w")
        self.min_area_slider = ttk.Scale(
            opts,
            from_=0,
            to=4,
            variable=self.var_min_area_log,
            orient=tk.HORIZONTAL,
            length=170,
            command=lambda value: self._on_min_area_change(float(value)),
        )
        self.min_area_slider.grid(row=2, column=1, sticky="w")
        ttk.Label(opts, text="Area Ratio").grid(row=2, column=2, sticky="w")
        ttk.Scale(opts, from_=0.1, to=1.0, variable=self.var_area_ratio, orient=tk.HORIZONTAL, length=140).grid(row=2, column=3, sticky="w")
        ttk.Label(opts, textvariable=self.var_camera_label).grid(row=2, column=4, sticky="w")
        self.camera_slider = ttk.Scale(
            opts,
            from_=1.0,
            to=0.1,
            variable=self.var_camera_threshold,
            orient=tk.HORIZONTAL,
            length=140,
            command=lambda value: self._on_camera_threshold_change(float(value)),
        )
        self.camera_slider.grid(row=2, column=5, sticky="w")

        ttk.Label(opts, text="ProRes").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=self.var_prores,
            values=["proxy", "lt", "standard", "hq", "4444", "4444xq"],
            width=12,
            state="readonly",
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(opts, text="Cutie Objects").grid(row=3, column=2, sticky="w")
        ttk.Spinbox(opts, from_=1, to=8, textvariable=self.var_cutie_objects, width=5).grid(row=3, column=3, sticky="w")
        ttk.Label(opts, text="Interp Factor").grid(row=3, column=4, sticky="w")
        ttk.Combobox(opts, textvariable=self.var_interp_factor, values=[2, 4, 8, 16], width=8, state="readonly").grid(
            row=3, column=5, sticky="w"
        )

        buttons = ttk.Frame(main)
        buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        ttk.Button(buttons, text="Track with Cutie", command=self._run_cutie_thread).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Process Video", command=self._process_video_thread).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Interpolate Last Output", command=self._interpolate_thread).pack(side=tk.LEFT, padx=4)

        preview = ttk.LabelFrame(main, text="Preview", padding=6)
        preview.grid(row=4, column=0, columnspan=4, sticky="nsew", pady=(6, 6))
        preview.grid_propagate(False)
        self.preview_container = preview
        self.preview_label = ttk.Label(preview, text="No preview")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        pctrl = ttk.Frame(preview)
        pctrl.pack(fill=tk.X)
        ttk.Button(pctrl, text="Previous", command=self._preview_prev).pack(side=tk.LEFT, padx=4)
        self.play_btn = ttk.Button(pctrl, text="Play", command=self._preview_toggle)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(pctrl, text="Next", command=self._preview_next).pack(side=tk.LEFT, padx=4)
        ttk.Label(pctrl, textvariable=self.var_frame_label).pack(side=tk.LEFT, padx=10)

        ttk.Progressbar(main, variable=self.var_progress, maximum=1.0).grid(row=5, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Label(main, textvariable=self.var_status, anchor="w").grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        for col in range(4):
            main.columnconfigure(col, weight=1)
        main.rowconfigure(4, weight=1)

        self.var_output.trace_add("write", self._on_output_dir_changed)
        self._on_motion_method_change()
        self._on_camera_threshold_change(float(self.var_camera_threshold.get()))
        self._sync_mask_state()
        self._sync_camera_state()
        self._sync_sift_state()

    def _add_path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, browse_cmd) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse", command=browse_cmd).grid(row=row, column=3, sticky="ew")

    def _workspace_root(self) -> str:
        out_dir = self.var_output.get().strip()
        return out_dir or str(self.paths.workspace_dir)

    def _on_output_dir_changed(self, *_args) -> None:
        self._sync_mask_state()

    @staticmethod
    def _log_to_pixels(log_value: float) -> int:
        return int(10 ** float(log_value))

    @staticmethod
    def _format_thousands(value: int) -> str:
        if abs(value) >= 10000:
            return f"{value / 1000:.1f}K"
        return str(value)

    def _on_min_area_change(self, value: float) -> None:
        px = self._log_to_pixels(value)
        self.var_min_area_label.set(f"Min Contour Area: {self._format_thousands(px)} px")

    def _on_motion_method_change(self) -> None:
        method = self.var_motion_method.get()
        if method == "Optical Flow":
            self.min_area_slider.configure(from_=0, to=5)
            self.var_min_area_log.set(4.0)
        else:
            self.min_area_slider.configure(from_=0, to=4)
            self.var_min_area_log.set(3.0)
        self._on_min_area_change(float(self.var_min_area_log.get()))

    def _on_camera_threshold_change(self, value: float) -> None:
        rounded = int(round(float(value), 2) * 100)
        self.var_camera_label.set(f"Cam Motion Sensitivity: {100 - rounded}%")

    def _refresh_mask_availability(self) -> bool:
        video = self.var_video.get().strip()
        workspace = self._workspace_root()
        available = bool(video) and mask_dir_for_video(workspace, video).exists()
        if available:
            self.use_masks_check.configure(state="normal")
        else:
            self.use_masks_check.configure(state="disabled")
            self.var_use_masks.set(False)
        return available

    def _sync_mask_state(self) -> None:
        available = self._refresh_mask_availability()
        if self.var_use_masks.get() and available:
            self.output_masked_check.configure(state="normal")
        else:
            self.output_masked_check.configure(state="disabled")
            self.var_output_masked.set(False)

    def _sync_camera_state(self) -> None:
        if self.var_camera.get():
            self.detector_combo.configure(state="readonly")
            self.camera_slider.configure(state="normal")
        else:
            self.detector_combo.configure(state="disabled")
            self.camera_slider.configure(state="disabled")
        self._sync_sift_state()

    def _sync_sift_state(self) -> None:
        if self.var_camera.get() and self.var_detector.get() == "SIFT":
            self.sift_slider.configure(state="normal")
        else:
            self.sift_slider.configure(state="disabled")

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")])
        if p:
            self.var_video.set(p)
            self.var_output.set(str(Path(p).parent))
            self._sync_mask_state()
            self._load_preview_video(p)

    def _browse_output(self) -> None:
        p = filedialog.askdirectory(title="Select Output Directory")
        if p:
            self.var_output.set(p)

    def _runtime_pythonpath_parts(self) -> list[str]:
        parts: list[str] = []

        app_dir = self.paths.app_dir
        if (app_dir / "ezframes").exists():
            parts.append(str(app_dir))
        if (app_dir / "src" / "ezframes").exists():
            parts.append(str(app_dir / "src"))

        env_dir = self.paths.runtime_env_dir
        if env_dir.exists():
            parts.append(str(env_dir))
        env_site_packages = env_dir / "Lib" / "site-packages"
        if env_site_packages.exists():
            parts.append(str(env_site_packages))

        return parts

    @staticmethod
    def _merge_pythonpath(parts: list[str], existing: str = "") -> str:
        merged: list[str] = []
        seen: set[str] = set()
        for part in [*parts, *existing.split(os.pathsep)]:
            p = str(part).strip()
            if not p:
                continue
            key = os.path.normcase(os.path.normpath(p))
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
        return os.pathsep.join(merged)

    def _cuda_support(self) -> tuple[bool, bool]:
        cv2_cuda = False
        torch_cuda = False

        try:
            cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            cv2_cuda = False

        try:
            import torch

            torch_cuda = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda = False

        return cv2_cuda, torch_cuda

    def _warn_if_no_cuda(self) -> None:
        cv2_cuda, torch_cuda = self._cuda_support()
        if not cv2_cuda and not torch_cuda:
            messagebox.showwarning("CUDA Not Detected", "CUDA acceleration was not detected. CPU fallback will be used.")
            return
        if torch_cuda and not cv2_cuda:
            self.var_status.set(
                "CUDA is available for PyTorch features (Cutie/RIFE), but OpenCV CUDA is unavailable. "
                "Optical-flow path will use CPU."
            )

    def _run_cutie_thread(self) -> None:
        threading.Thread(target=self._run_cutie, daemon=True).start()

    @staticmethod
    def _python_can_import_module(python_exe: str, module_name: str, env: dict[str, str] | None = None) -> bool:
        try:
            completed = subprocess.run(
                [python_exe, "-c", f"import {module_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                shell=False,
                env=env,
            )
            return completed.returncode == 0
        except Exception:
            return False

    @classmethod
    def _python_can_import_modules(
        cls, python_exe: str, module_names: list[str], env: dict[str, str] | None = None
    ) -> bool:
        for module_name in module_names:
            if not cls._python_can_import_module(python_exe, module_name, env=env):
                return False
        return True

    def _build_cutie_env(self, src_root: Path | None = None) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath_parts = self._runtime_pythonpath_parts()
        if src_root is not None:
            pythonpath_parts.append(str(src_root))
        env["PYTHONPATH"] = self._merge_pythonpath(pythonpath_parts, env.get("PYTHONPATH", ""))
        env.setdefault("EZFRAMES_SOURCE_ROOT", str((src_root.parent if src_root else self.paths.install_root)))
        return env

    def _select_cutie_python(self) -> str:
        if self._cutie_python:
            return self._cutie_python

        candidates: list[Path] = []
        env_cutie_python = os.environ.get("EZFRAMES_CUTIE_PYTHON", "").strip()
        if env_cutie_python:
            candidates.append(Path(env_cutie_python))

        for root in self.paths.source_roots():
            candidates.append(root / "cutievenv" / "Scripts" / "python.exe")

        candidates.append(self.paths.runtime_env_dir / "Scripts" / "python.exe")
        candidates.append(self.paths.runtime_python_dir / "python.exe")
        candidates.append(Path(sys.executable))
        candidates.append(self.paths.python_exe)

        src_root = Path(__file__).resolve().parents[2]
        env = self._build_cutie_env(src_root=src_root)

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            c_str = str(candidate)
            if c_str in seen:
                continue
            seen.add(c_str)

            # Native cutie-lite path requires these, not Qt.
            if self._python_can_import_modules(c_str, ["torch", "cv2", "omegaconf"], env=env):
                self._cutie_python = c_str
                return c_str

        raise RuntimeError(
            "No Python interpreter with required Cutie-lite dependencies was found "
            "(torch, cv2, omegaconf). "
            "Set EZFRAMES_CUTIE_PYTHON to a valid interpreter."
        )

    def _run_cutie(self) -> None:
        video = self.var_video.get().strip()
        if not video:
            self.root.after(0, lambda: messagebox.showerror("Error", "Select a video first."))
            return
        workspace_root = self._workspace_root()
        video_workspace = video_workspace_dir(workspace_root, video)
        output_mask_dir = str(video_workspace / "binary_masks")
        status_file = str(video_workspace / "cutie_status.json")
        try:
            cutie_python = self._select_cutie_python()
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda m=msg: messagebox.showerror("Cutie Failed", m))
            return

        src_root = Path(__file__).resolve().parents[2]
        env = self._build_cutie_env(src_root=src_root)

        cmd = [
            cutie_python,
            "-m",
            "ezframes.cutie_lite",
            "--video",
            video,
            "--num-objects",
            str(self.var_cutie_objects.get()),
            "--workspace",
            str(video_workspace),
            "--output-mask-dir",
            output_mask_dir,
            "--status-file",
            status_file,
        ]
        self.root.after(0, lambda vw=str(video_workspace): self.var_status.set(f"Launching Cutie-lite ({vw})..."))
        rc = subprocess.call(cmd, shell=False, env=env)
        if rc == 0:
            self.root.after(0, lambda: self.var_status.set(f"Cutie-lite complete. Masks: {output_mask_dir}"))
            self.root.after(0, self._sync_mask_state)
        else:
            detail = f"Cutie process exited with code {rc}."
            try:
                status_data = json.loads(Path(status_file).read_text(encoding="utf-8"))
                err = str(status_data.get("error", "")).strip()
                if err:
                    detail = f"{detail}\n\n{err}"
            except Exception:
                pass
            self.root.after(0, lambda d=detail: messagebox.showerror("Cutie Failed", d))

    def _process_video_thread(self) -> None:
        threading.Thread(target=self._process_video, daemon=True).start()

    @staticmethod
    def _min_area_pixels(log_value: float) -> float:
        return float(EzFramesUI._log_to_pixels(log_value))

    def _build_frame(self, frame: cv2.typing.MatLike, contour, mask, debug: bool, use_masks: bool, alpha_masked: bool):
        out = frame.copy()
        if contour is not None and debug:
            out = draw_contour(out, contour, cutie_mask=mask)
        # Legacy parity: "Use Cutie Masks" gates detection only.
        # Output pixels are masked only when explicit masked-alpha output is enabled.
        if alpha_masked and mask is not None:
            out = cv2.bitwise_and(out, out, mask=mask)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)
            out[:, :, 3] = mask
        return out

    def _process_video(self) -> None:
        video = self.var_video.get().strip()
        out_dir = self.var_output.get().strip()
        workspace = self._workspace_root()
        if not video or not out_dir:
            self.root.after(0, lambda: messagebox.showerror("Error", "Video and output directory are required."))
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        opts = MotionOptions(
            min_area=self._min_area_pixels(self.var_min_area_log.get()),
            method=self.var_motion_method.get(),
            area_ratio_threshold=float(self.var_area_ratio.get()),
            use_cuda=self._cuda_support()[0],
        )
        use_masks = bool(self.var_use_masks.get())
        alpha_masked = bool(self.var_output_masked.get()) and use_masks
        output_images = bool(self.var_output_images.get())
        camera_filter = bool(self.var_camera.get())
        debug = bool(self.var_debug.get())

        detector = self.var_detector.get()
        sift_scale = float(self.var_sift_scale.get())
        camera_thr = float(self.var_camera_threshold.get())
        prores = self.var_prores.get()

        base = Path(video).stem
        mask_dir = mask_dir_for_video(workspace, video)
        out_video = str(Path(out_dir) / f"{base}.mov")
        out_images = Path(out_dir) / base
        out_homography = Path(out_dir) / f"{base}_homography"
        if use_masks and not mask_dir.exists():
            use_masks = False
            alpha_masked = False
            self.root.after(
                0,
                lambda: self.var_status.set(f"Mask directory not found ({mask_dir}). Processing without masks."),
            )

        if out_images.exists() and output_images:
            shutil.rmtree(out_images, ignore_errors=True)
        if out_homography.exists() and debug:
            shutil.rmtree(out_homography, ignore_errors=True)

        try:
            cap, width, height, fps, total = open_capture(video)
            ffmpeg = resolve_ffmpeg(self.paths)
            self.root.after(0, lambda: self.var_status.set(f"Processing {total} frames..."))
            self.root.after(0, lambda: self.var_progress.set(0.0))

            ret, previous = cap.read()
            if not ret:
                raise RuntimeError("Could not read first frame.")

            frame_idx = 1
            kept = 0
            written = 0
            prev_area = 0.0
            pix_fmt = "bgra" if alpha_masked else "bgr24"
            ctx = (
                FFmpegWriter(ffmpeg_path=ffmpeg, output_path=out_video, width=width, height=height, fps=fps, pix_fmt=pix_fmt, profile=prores)
                if not output_images
                else _NullContext()
            )
            if output_images:
                out_images.mkdir(parents=True, exist_ok=True)

            def _write(frame):
                nonlocal written
                if output_images:
                    cv2.imwrite(str(out_images / f"{base}_{written:04d}.png"), frame)
                else:
                    writer.write(frame)
                written += 1

            with ctx as writer:
                first_mask = load_mask(mask_dir, frame_idx) if use_masks else None
                _write(self._build_frame(previous, None, first_mask, False, use_masks, alpha_masked))
                kept += 1

                while cap.isOpened():
                    ret, current = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    cutie_mask = load_mask(mask_dir, frame_idx) if use_masks else None

                    if camera_filter:
                        cam = detect_camera_motion(
                            frame1=previous,
                            frame2=current,
                            mask=cutie_mask,
                            feature_detector=detector,
                            sift_scale=sift_scale,
                            inlier_ratio_threshold=camera_thr,
                        )
                        if cam.is_camera_motion:
                            if debug and cam.homography is not None and len(cam.good_matches) >= 4:
                                out_homography.mkdir(parents=True, exist_ok=True)
                                vis = visualize_homography(cam.vis_frame1, cam.vis_frame2, cam.homography, cam.good_matches, cam.kp1, cam.kp2)
                                cv2.imwrite(str(out_homography / f"homography_{frame_idx:04d}.png"), vis)
                            previous = current
                            self.root.after(0, lambda p=min(1.0, frame_idx / max(1, total)): self.var_progress.set(p))
                            continue

                    contour, prev_area = detect_motion(previous, current, options=opts, previous_area=prev_area, cutie_mask=cutie_mask)
                    if contour is not None:
                        _write(self._build_frame(current, contour, cutie_mask, debug, use_masks, alpha_masked))
                        kept += 1
                    previous = current

                    if frame_idx % 20 == 0:
                        p = min(1.0, frame_idx / max(1, total))
                        self.root.after(0, lambda p=p: self.var_progress.set(p))
                        self.root.after(0, lambda i=frame_idx, k=kept: self.var_status.set(f"Processing {i}/{total} | kept {k}"))

            cap.release()

            self.root.after(0, lambda: self.var_progress.set(1.0))
            if output_images:
                self.last_output_path = str(out_images)
                self.root.after(0, lambda: self.var_status.set(f"Done. Kept {kept} frames to {out_images}"))
            else:
                self.last_output_path = out_video
                self.root.after(0, lambda: self.var_status.set(f"Done. Kept {kept} frames to {out_video}"))
                self.root.after(0, lambda: self._load_preview_video(out_video))
        except Exception as exc:
            log.exception("Process video failed: %s", exc)
            msg = str(exc)
            self.root.after(0, lambda m=msg: messagebox.showerror("Processing Failed", m))

    def _interpolate_thread(self) -> None:
        threading.Thread(target=self._interpolate, daemon=True).start()

    def _interpolate(self) -> None:
        if not self.last_output_path:
            self.root.after(0, lambda: messagebox.showerror("Error", "No processed output available."))
            return
        if Path(self.last_output_path).is_dir():
            self.root.after(0, lambda: messagebox.showerror("Error", "Interpolation needs video output, not image sequence."))
            return
        try:
            profiles = ["proxy", "lt", "standard", "hq", "4444", "4444xq"]
            quality = profiles.index(self.var_prores.get()) + 1
            sf = int(self.var_interp_factor.get())
            output = str(Path(self.last_output_path).with_name(Path(self.last_output_path).stem + "_interpolated.mov"))
            self.root.after(0, lambda: self.var_status.set(f"Interpolating x{sf}..."))

            self.root.after(0, lambda: self.var_progress.set(0.0))

            def _progress(current: int, total: int) -> None:
                p = min(1.0, float(current) / max(1, int(total)))
                self.root.after(0, lambda p=p: self.var_progress.set(p))
                self.root.after(0, lambda c=current, t=total: self.var_status.set(f"Interpolating {c}/{t} frames..."))

            self.interpolator.interpolate(
                InterpolationRequest(input_path=self.last_output_path, output_path=output, sf=sf, prores_quality=quality),
                progress_callback=_progress,
            )
            self.root.after(0, lambda: self.var_progress.set(1.0))
            self.root.after(0, lambda: self.var_status.set(f"Interpolation complete: {output}"))
            self.root.after(0, lambda: self._load_preview_video(output))
        except Exception as exc:
            log.exception("Interpolation failed: %s", exc)
            msg = str(exc)
            self.root.after(0, lambda m=msg: messagebox.showerror("Interpolation Failed", m))

    def _load_preview_video(self, video_path: str) -> None:
        self._stop_preview()
        if self.preview_cap is not None:
            self.preview_cap.release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.var_frame_label.set("Preview unavailable")
            self.preview_cap = None
            return
        self.preview_cap = cap
        self.preview_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.preview_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        self.preview_frame = 0
        self._show_frame(0, seek=True)

    def _show_frame(self, idx: int | None, seek: bool = True) -> None:
        if self.preview_cap is None:
            return
        total = max(1, self.preview_total_frames)

        if seek:
            if idx is None:
                idx = self.preview_frame
            idx = max(0, min(total - 1, idx))
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.preview_cap.read()
            if not ok:
                return
            self.preview_frame = idx
        else:
            ok, frame = self.preview_cap.read()
            if not ok:
                # Loop playback from start.
                self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.preview_cap.read()
                if not ok:
                    return
            pos = int(self.preview_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.preview_frame = max(0, min(total - 1, pos - 1))

        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        container_w = max(1, self.preview_container.winfo_width())
        container_h = max(1, self.preview_container.winfo_height())
        if container_w <= 1 or container_h <= 1:
            target_w, target_h = 960, 360
        else:
            # Reserve vertical space for controls/labels under the preview image.
            target_w = max(320, container_w - 20)
            target_h = max(180, container_h - 90)

        src_h, src_w = rgb.shape[:2]
        scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
        draw_w = max(1, int(src_w * scale))
        draw_h = max(1, int(src_h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (draw_w, draw_h), interpolation=interp)
        img = Image.fromarray(resized)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_photo, text="")
        frame_num = self.preview_frame + 1
        self.var_frame_label.set(f"Frame: {frame_num}/{self.preview_total_frames} | FPS: {math.ceil(self.preview_fps * 100) / 100}")

    def _preview_prev(self) -> None:
        self._stop_preview()
        self._show_frame(self.preview_frame - 1, seek=True)

    def _preview_next(self) -> None:
        self._stop_preview()
        next_idx = self.preview_frame + 1
        if next_idx >= self.preview_total_frames:
            next_idx = 0
        self._show_frame(next_idx, seek=True)

    def _preview_toggle(self) -> None:
        self.preview_playing = not self.preview_playing
        self.play_btn.configure(text="Pause" if self.preview_playing else "Play")
        if self.preview_playing:
            self.preview_next_tick_s = time.perf_counter()
            self._preview_tick()
        else:
            self._stop_preview(cancel_only=True)

    def _preview_tick(self) -> None:
        if not self.preview_playing:
            return

        self._show_frame(None, seek=False)
        interval_s = 1.0 / max(1.0, self.preview_fps)
        now = time.perf_counter()

        if self.preview_next_tick_s is None:
            self.preview_next_tick_s = now
        self.preview_next_tick_s += interval_s
        if self.preview_next_tick_s < now:
            self.preview_next_tick_s = now

        delay = int(max(1, (self.preview_next_tick_s - now) * 1000))
        self.preview_job = self.root.after(delay, self._preview_tick)

    def _stop_preview(self, cancel_only: bool = False) -> None:
        self.preview_playing = False
        self.preview_next_tick_s = None
        self.play_btn.configure(text="Play")
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except Exception:
                pass
            self.preview_job = None
        if cancel_only:
            return

    def _on_close(self) -> None:
        self._stop_preview()
        if self.preview_cap is not None:
            self.preview_cap.release()
        self.root.destroy()


def main() -> int:
    if os.environ.get("EZFRAMES_APP_AUTH_OK", "") != "1":
        auth_code = enforce_authorized_launch(interactive_error=True, consume_ticket=True)
        if auth_code != 0:
            return auth_code
        os.environ["EZFRAMES_APP_AUTH_OK"] = "1"

    paths = AppPaths.default()
    paths.ensure_layout()
    configure_logging(paths.logs_dir)

    root = tk.Tk()
    EzFramesUI(root, paths)
    root.mainloop()
    return 0

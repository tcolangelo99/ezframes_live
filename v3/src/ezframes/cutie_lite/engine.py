from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
import threading
import time
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

from ezframes.common.config import AppPaths


log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EzFrames native Cutie-lite tracker (no Qt)")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--workspace", required=True, help="Per-video workspace directory")
    p.add_argument("--output-mask-dir", required=True, help="Binary mask output directory")
    p.add_argument("--num-objects", type=int, default=1, help="Maximum object IDs for prompting")
    p.add_argument("--status-file", default="", help="Optional status JSON path")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--mem-every", type=int, default=None, help="Optional memory interval override")
    p.add_argument(
        "--max-internal-size",
        type=int,
        default=None,
        help="Optional CUTIE max_internal_size override (short side)",
    )
    p.add_argument(
        "--max-overall-size",
        type=int,
        default=None,
        help="Optional frame extraction max_overall_size override (short side)",
    )
    return p


def _write_status(path: str, payload: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _has_cutie_tree(candidate: Path) -> bool:
    return (candidate / "cutie").is_dir() and (candidate / "gui").is_dir()


def _dedupe_paths(candidates: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_cutie_root_from_imports() -> Path | None:
    cutie_spec = importlib.util.find_spec("cutie")
    gui_spec = importlib.util.find_spec("gui")
    if cutie_spec is None or gui_spec is None:
        return None
    if not cutie_spec.origin or not gui_spec.origin:
        return None

    cutie_pkg = Path(cutie_spec.origin).resolve().parent
    gui_pkg = Path(gui_spec.origin).resolve().parent

    # Prefer a shared root containing both packages, e.g. app/src or site-packages.
    shared = cutie_pkg.parent
    if gui_pkg.parent == shared and _has_cutie_tree(shared):
        return shared
    if _has_cutie_tree(cutie_pkg.parent):
        return cutie_pkg.parent
    return None


def _resolve_cutie_root(paths: AppPaths) -> Path:
    env_root = os.environ.get("EZFRAMES_CUTIE_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    # Installed app layouts.
    candidates.extend(
        [
            paths.app_dir / "src",
            paths.app_dir,
            paths.install_root / "app" / "src",
            paths.install_root / "app",
        ]
    )

    for root in paths.source_roots():
        candidates.append(root / "app" / "src")
        candidates.append(root / "app")
        candidates.append(root / "Cutietop")
        candidates.append(root)

    for candidate in _dedupe_paths(candidates):
        if _has_cutie_tree(candidate):
            return candidate

    imported_root = _resolve_cutie_root_from_imports()
    if imported_root is not None:
        return imported_root

    raise RuntimeError(
        "Could not locate Cutie source root (expected folders: cutie/, gui/ under app/, app/src/, or Cutietop/)."
    )


def _prepare_cutie_import_paths(cutie_root: Path) -> None:
    for candidate in (cutie_root, cutie_root.parent):
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _resolve_weight_file(name: str, paths: AppPaths, cutie_root: Path, env_var: str) -> Path:
    env_value = os.environ.get(env_var, "").strip()
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value))

    candidates.extend(
        [
            cutie_root / "weights" / name,
            paths.models_dir / name,
            paths.install_root / "weights" / name,
        ]
    )

    for root in paths.source_roots():
        candidates.extend(
            [
                root / "weights" / name,
                root / "models" / name,
                root / "Cutietop" / "weights" / name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Required weight file not found: {name} (override with {env_var}).")


def _select_device(requested: str) -> str:
    import torch

    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _build_cfg(
    cutie_root: Path,
    video_path: Path,
    workspace: Path,
    num_objects: int,
    device: str,
    cutie_weights: Path,
    ritm_weights: Path,
    mem_every: int | None,
    max_internal_size: int | None,
    max_overall_size: int | None,
):
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.load(str(cutie_root / "cutie" / "config" / "gui_config.yaml"))
    model_cfg = OmegaConf.load(str(cutie_root / "cutie" / "config" / "model" / "base.yaml"))

    with open_dict(cfg):
        cfg.model = model_cfg
        if "defaults" in cfg:
            del cfg["defaults"]

        cfg.images = None
        cfg.video = str(video_path)
        cfg.workspace = str(workspace)
        cfg.num_objects = int(num_objects)
        cfg.device = str(device)
        cfg.amp = bool(device == "cuda")
        cfg.weights = str(cutie_weights)
        cfg.ritm_weights = str(ritm_weights)

        if mem_every is not None:
            cfg.mem_every = int(mem_every)
        if max_internal_size is not None:
            cfg.max_internal_size = int(max_internal_size)
        if max_overall_size is not None:
            cfg.max_overall_size = int(max_overall_size)

        # Reduce peak VRAM spikes in long propagations while keeping quality close to defaults.
        if str(device) == "cuda":
            cfg.chunk_size = int(os.environ.get("EZFRAMES_CUTIE_CHUNK_SIZE", "4"))
    return cfg


class CutieLiteSession:
    def __init__(self, cfg, output_mask_dir: Path):
        import cv2
        import numpy as np
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import Image, ImageTk

        from cutie.inference.inference_core import InferenceCore
        from cutie.model.cutie import CUTIE
        import gui.click_controller as click_controller_module
        from gui.click_controller import ClickController
        from gui.interaction import ClickInteraction
        from gui.interactive_utils import get_visualization, image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask
        from gui.resource_manager import ResourceManager
        import gui.resource_manager as resource_manager_module

        self.cv2 = cv2
        self.np = np
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.Image = Image
        self.ImageTk = ImageTk

        # Avoid Cython runtime compilation from RITM dist maps on end-user systems.
        try:
            load_is_model = click_controller_module.utils.load_is_model
            if not getattr(load_is_model, "_ezframes_cpu_distmaps_disabled", False):
                def _load_is_model_no_cpu_distmaps(*args, **kwargs):
                    kwargs["cpu_dist_maps"] = False
                    return load_is_model(*args, **kwargs)

                _load_is_model_no_cpu_distmaps._ezframes_cpu_distmaps_disabled = True
                click_controller_module.utils.load_is_model = _load_is_model_no_cpu_distmaps
        except Exception as patch_exc:
            log.warning("Could not patch ClickController dist-map mode: %s", patch_exc)

        self.CUTIE = CUTIE
        self.InferenceCore = InferenceCore
        self.ClickController = ClickController
        self.ClickInteraction = ClickInteraction
        self.get_visualization = get_visualization
        self.image_to_torch = image_to_torch
        self.index_numpy_to_one_hot_torch = index_numpy_to_one_hot_torch
        self.torch_prob_to_numpy_mask = torch_prob_to_numpy_mask
        self.ResourceManager = ResourceManager
        self.resource_manager_module = resource_manager_module

        self.cfg = cfg
        self.output_mask_dir = output_mask_dir

        import torch
        self.torch = torch

        self.root = self.tk.Tk()
        self.root.title("EzFrames Cutie Lite")
        self.root.geometry("760x170")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

        self.loading_status_var = self.tk.StringVar(value="Initializing Cutie-lite...")
        self.loading_progress_var = self.tk.DoubleVar(value=0.0)
        self.loading_frame = None
        self.loading_bar = None
        self._loading_mode = "indeterminate"
        self._loading_total_hint = self._read_video_total_frames(cfg.video)
        self._loading_last_pump = 0.0
        self._build_loading_ui()
        self._set_loading_stage("Loading segmentation model...")

        # Match original Cutie demo behavior to avoid autograd graph buildup.
        torch.set_grad_enabled(False)

        self.cutie = self.CUTIE(cfg).eval().to(cfg.device)

        self._set_loading_stage("Loading Cutie weights...")
        model_weights = torch.load(cfg.weights, map_location=cfg.device)
        self.cutie.load_weights(model_weights)

        self._set_loading_stage("Loading click model...")
        self.click_ctrl = self.ClickController(cfg.ritm_weights, device=cfg.device)
        self._set_loading_stage("Preparing video frames...")
        self.res_man = self._build_resource_manager_with_loading()

        self.total_frames = int(self.res_man.T)
        self.h = int(self.res_man.h)
        self.w = int(self.res_man.w)
        self.num_objects = int(cfg.num_objects)

        self.current_idx = 0
        self.current_obj = 1
        self.playing = False
        self.play_job = None
        self.play_next_tick = None
        self.fps = self._read_video_fps(cfg.video)

        self.masks_by_frame: dict[int, np.ndarray] = {}
        self.active_obj_ids: list[int] = []
        self.interaction = None
        self.interaction_frame = -1
        self.interaction_obj = -1
        self.mask_lock = threading.Lock()

        self.propagating = False
        self.propagation_thread = None
        self.exported_count = 0
        self.closed = False

        self.display_scale = 1.0
        self.display_w = self.w
        self.display_h = self.h
        self.frame_photo = None

        self._teardown_loading_ui()
        self.root.title("EzFrames Cutie Lite")
        self.root.geometry("1180x860")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Left>", lambda _e: self._prev_frame())
        self.root.bind("<Right>", lambda _e: self._next_frame())
        self.root.bind("<space>", lambda _e: self._toggle_play())

        self.status_var = self.tk.StringVar(value="Ready. Click to add prompts.")
        self.frame_var = self.tk.StringVar(value="")
        self.progress_var = self.tk.DoubleVar(value=0.0)
        self.obj_var = self.tk.IntVar(value=1)

        self._build_ui()
        self._refresh_frame()

    def _build_loading_ui(self) -> None:
        self.loading_frame = self.ttk.Frame(self.root, padding=14)
        self.loading_frame.pack(fill=self.tk.BOTH, expand=True)
        self.ttk.Label(self.loading_frame, text="Cutie-lite is preparing this video.").pack(anchor="w", pady=(0, 4))
        self.ttk.Label(self.loading_frame, textvariable=self.loading_status_var).pack(anchor="w", pady=(0, 8))
        self.loading_bar = self.ttk.Progressbar(
            self.loading_frame,
            variable=self.loading_progress_var,
            mode="indeterminate",
            maximum=max(1, self._loading_total_hint),
        )
        self.loading_bar.pack(fill=self.tk.X)
        self.loading_bar.start(12)
        self._pump_loading_ui(force=True)

    def _pump_loading_ui(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._loading_last_pump) < 0.05:
            return
        self._loading_last_pump = now
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            pass

    def _set_loading_stage(self, message: str) -> None:
        self.loading_status_var.set(message)
        if self.loading_bar is not None and self._loading_mode != "indeterminate":
            self.loading_bar.configure(mode="indeterminate")
            self.loading_bar.start(12)
            self._loading_mode = "indeterminate"
        self._pump_loading_ui(force=True)

    def _set_loading_progress(self, current: int, total: int) -> None:
        current = max(0, int(current))
        total = max(0, int(total))
        if total > 0:
            if self.loading_bar is not None and self._loading_mode != "determinate":
                self.loading_bar.stop()
                self.loading_bar.configure(mode="determinate", maximum=max(1, total))
                self._loading_mode = "determinate"
            if self.loading_bar is not None:
                self.loading_bar.configure(maximum=max(1, total))
            shown = min(current, total)
            self.loading_progress_var.set(shown)
            self.loading_status_var.set(f"Preparing video frames... {shown}/{total}")
        else:
            self.loading_status_var.set(f"Preparing video frames... {current}")
        self._pump_loading_ui(force=(total > 0 and current >= total))

    def _make_loading_tqdm_factory(self):
        session = self

        class _LoadingTqdm:
            def __init__(self, iterable=None, total=None):
                self.iterable = iterable
                if total is None and iterable is not None:
                    try:
                        total = len(iterable)
                    except Exception:
                        total = None
                if total is None or int(total) <= 0:
                    total = session._loading_total_hint if session._loading_total_hint > 0 else 0
                self.total = int(total) if total is not None else 0
                self.n = 0
                session._set_loading_stage("Preparing video frames...")
                if self.total > 0:
                    session._set_loading_progress(0, self.total)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                if self.total > 0:
                    session._set_loading_progress(self.total, self.total)
                return False

            def update(self, n=1):
                self.n += int(n)
                session._set_loading_progress(self.n, self.total)

            def __iter__(self):
                if self.iterable is None:
                    return
                for item in self.iterable:
                    yield item
                    self.n += 1
                    session._set_loading_progress(self.n, self.total)
                if self.total > 0:
                    session._set_loading_progress(self.total, self.total)

            def close(self):
                return None

        def _factory(iterable=None, *args, **kwargs):
            total = kwargs.get("total")
            if iterable is None and args:
                iterable = args[0]
            return _LoadingTqdm(iterable=iterable, total=total)

        return _factory

    def _build_resource_manager_with_loading(self):
        original_tqdm = self.resource_manager_module.tqdm
        self.resource_manager_module.tqdm = self._make_loading_tqdm_factory()
        try:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with redirect_stdout(sink), redirect_stderr(sink):
                    return self.ResourceManager(self.cfg)
        finally:
            self.resource_manager_module.tqdm = original_tqdm

    def _teardown_loading_ui(self) -> None:
        if self.loading_bar is not None and self._loading_mode == "indeterminate":
            try:
                self.loading_bar.stop()
            except Exception:
                pass
        if self.loading_frame is not None:
            try:
                self.loading_frame.destroy()
            except Exception:
                pass
        self.loading_bar = None
        self.loading_frame = None

    def _read_video_total_frames(self, video_path: str) -> int:
        cap = self.cv2.VideoCapture(video_path)
        total = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return max(0, total)

    def _read_video_fps(self, video_path: str) -> float:
        cap = self.cv2.VideoCapture(video_path)
        fps = float(cap.get(self.cv2.CAP_PROP_FPS) or 24.0)
        cap.release()
        return fps if fps > 0 else 24.0

    def _build_ui(self) -> None:
        main = self.ttk.Frame(self.root, padding=10)
        main.pack(fill=self.tk.BOTH, expand=True)

        viewer = self.ttk.LabelFrame(main, text="Prompt / Preview", padding=6)
        viewer.pack(fill=self.tk.BOTH, expand=True)
        viewer.pack_propagate(False)
        self.image_label = self.ttk.Label(viewer, text="")
        self.image_label.pack(fill=self.tk.BOTH, expand=True)
        self.image_label.bind("<Button-1>", lambda e: self._on_click(e, is_neg=False))
        self.image_label.bind("<Button-3>", lambda e: self._on_click(e, is_neg=True))

        control_row_1 = self.ttk.Frame(main)
        control_row_1.pack(fill=self.tk.X, pady=(8, 4))
        self.btn_prev = self.ttk.Button(control_row_1, text="Previous", command=self._prev_frame)
        self.btn_prev.pack(side=self.tk.LEFT, padx=4)
        self.btn_play = self.ttk.Button(control_row_1, text="Play", command=self._toggle_play)
        self.btn_play.pack(side=self.tk.LEFT, padx=4)
        self.btn_next = self.ttk.Button(control_row_1, text="Next", command=self._next_frame)
        self.btn_next.pack(side=self.tk.LEFT, padx=4)
        self.ttk.Label(control_row_1, textvariable=self.frame_var).pack(side=self.tk.LEFT, padx=12)

        control_row_2 = self.ttk.Frame(main)
        control_row_2.pack(fill=self.tk.X, pady=(0, 4))
        self.ttk.Label(control_row_2, text="Object ID").pack(side=self.tk.LEFT, padx=4)
        self.obj_spin = self.ttk.Spinbox(
            control_row_2, from_=1, to=max(1, self.num_objects), textvariable=self.obj_var, width=6
        )
        self.obj_spin.pack(side=self.tk.LEFT, padx=4)
        self.obj_spin.bind("<KeyRelease>", lambda _e: self._on_object_change())
        self.obj_spin.bind("<<Increment>>", lambda _e: self._on_object_change())
        self.obj_spin.bind("<<Decrement>>", lambda _e: self._on_object_change())
        self.ttk.Button(control_row_2, text="Clear Object", command=self._clear_current_object).pack(
            side=self.tk.LEFT, padx=4
        )
        self.ttk.Button(control_row_2, text="Reset Prompts", command=self._reset_session).pack(
            side=self.tk.LEFT, padx=4
        )

        control_row_3 = self.ttk.Frame(main)
        control_row_3.pack(fill=self.tk.X, pady=(0, 4))
        self.btn_backward = self.ttk.Button(
            control_row_3, text="Propagate Backward", command=lambda: self._start_propagation("backward")
        )
        self.btn_backward.pack(side=self.tk.LEFT, padx=4)
        self.btn_forward = self.ttk.Button(
            control_row_3, text="Propagate Forward", command=lambda: self._start_propagation("forward")
        )
        self.btn_forward.pack(side=self.tk.LEFT, padx=4)
        self.btn_export = self.ttk.Button(control_row_3, text="Export Masks", command=self._export_masks)
        self.btn_export.pack(side=self.tk.LEFT, padx=4)
        self.ttk.Button(control_row_3, text="Close", command=self._on_close).pack(side=self.tk.RIGHT, padx=4)

        self.ttk.Progressbar(main, variable=self.progress_var, maximum=1.0).pack(fill=self.tk.X, pady=(6, 0))
        self.ttk.Label(main, textvariable=self.status_var, anchor="w").pack(fill=self.tk.X, pady=(4, 0))

    def _set_interaction_none(self) -> None:
        self.interaction = None
        self.interaction_frame = -1
        self.interaction_obj = -1
        self.click_ctrl.unanchor()

    def _current_mask(self) -> "np.ndarray":
        with self.mask_lock:
            existing = self.masks_by_frame.get(self.current_idx)
            if existing is None:
                return self.np.zeros((self.h, self.w), dtype=self.np.uint8)
            return existing.copy()

    def _set_current_mask(self, mask: "np.ndarray") -> None:
        with self.mask_lock:
            self.masks_by_frame[self.current_idx] = mask.astype(self.np.uint8)

        obj_ids = sorted(int(v) for v in self.np.unique(mask) if int(v) > 0)
        self.active_obj_ids = sorted(set(self.active_obj_ids) | set(obj_ids))

    def _on_object_change(self) -> None:
        try:
            value = int(self.obj_var.get())
        except Exception:
            value = 1
        value = max(1, min(self.num_objects, value))
        self.obj_var.set(value)
        self.current_obj = value
        self._set_interaction_none()
        self._refresh_frame()

    def _map_click_to_frame(self, event_x: int, event_y: int) -> tuple[int, int] | None:
        if self.display_w <= 0 or self.display_h <= 0:
            return None
        x = int(event_x / max(1e-6, self.display_scale))
        y = int(event_y / max(1e-6, self.display_scale))
        if not (0 <= x < self.w and 0 <= y < self.h):
            return None
        return x, y

    def _on_click(self, event, is_neg: bool) -> None:
        if self.propagating:
            return
        mapped = self._map_click_to_frame(event.x, event.y)
        if mapped is None:
            return
        x, y = mapped

        frame = self.res_man.get_image(self.current_idx)
        current_mask = self._current_mask()
        with self.torch.inference_mode():
            if self.interaction is None or self.interaction_frame != self.current_idx or self.interaction_obj != self.current_obj:
                prev_prob = self.index_numpy_to_one_hot_torch(current_mask, self.num_objects + 1).to(
                    self.cfg.device, non_blocking=True
                )
                image_torch = self.image_to_torch(frame, device=self.cfg.device)
                self.interaction = self.ClickInteraction(
                    image_torch, prev_prob, (self.h, self.w), self.click_ctrl, self.current_obj
                )
                self.interaction_frame = self.current_idx
                self.interaction_obj = self.current_obj

            self.interaction.push_point(x, y, is_neg=is_neg)
            out_prob = self.interaction.predict().to(self.cfg.device).detach()

        new_mask = self.torch_prob_to_numpy_mask(out_prob).astype(self.np.uint8)
        self._set_current_mask(new_mask)
        self.status_var.set(
            f"Prompt updated on frame {self.current_idx + 1}. Object {self.current_obj}. "
            f"{'Remove' if is_neg else 'Add'} click at ({x}, {y})."
        )
        self._refresh_frame()

    def _clear_current_object(self) -> None:
        if self.propagating:
            return
        mask = self._current_mask()
        mask[mask == self.current_obj] = 0
        self._set_current_mask(mask)
        self._set_interaction_none()
        self.status_var.set(f"Cleared object {self.current_obj} on frame {self.current_idx + 1}.")
        self._refresh_frame()

    def _reset_session(self) -> None:
        if self.propagating:
            return
        with self.mask_lock:
            self.masks_by_frame.clear()
        self.active_obj_ids = []
        self._set_interaction_none()
        self.progress_var.set(0.0)
        self.status_var.set("All prompts/tracking reset.")
        self._refresh_frame()

    def _prev_frame(self) -> None:
        if self.propagating:
            return
        self._stop_playback()
        self.current_idx = (self.current_idx - 1) % self.total_frames
        self._set_interaction_none()
        self._refresh_frame()

    def _next_frame(self) -> None:
        if self.propagating:
            return
        self._stop_playback()
        self.current_idx = (self.current_idx + 1) % self.total_frames
        self._set_interaction_none()
        self._refresh_frame()

    def _toggle_play(self) -> None:
        if self.propagating:
            return
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.play_next_tick = time.perf_counter()
            self._play_tick()
        else:
            self._stop_playback(cancel_only=True)

    def _play_tick(self) -> None:
        if not self.playing:
            return
        self.current_idx = (self.current_idx + 1) % self.total_frames
        self._set_interaction_none()
        self._refresh_frame()

        interval = 1.0 / max(1.0, self.fps)
        now = time.perf_counter()
        if self.play_next_tick is None:
            self.play_next_tick = now
        self.play_next_tick += interval
        if self.play_next_tick < now:
            self.play_next_tick = now
        delay_ms = int(max(1, (self.play_next_tick - now) * 1000))
        self.play_job = self.root.after(delay_ms, self._play_tick)

    def _stop_playback(self, cancel_only: bool = False) -> None:
        self.playing = False
        self.play_next_tick = None
        self.btn_play.configure(text="Play")
        if self.play_job is not None:
            try:
                self.root.after_cancel(self.play_job)
            except Exception:
                pass
            self.play_job = None
        if cancel_only:
            return

    def _set_ui_for_propagation(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for btn in (self.btn_prev, self.btn_play, self.btn_next, self.btn_backward, self.btn_forward, self.btn_export):
            btn.configure(state=state)
        self.obj_spin.configure(state=state)

    def _start_propagation(self, direction: str) -> None:
        if self.propagating:
            return
        self._stop_playback()

        anchor_idx = self.current_idx
        anchor_mask = self._current_mask()
        active_obj_ids = sorted(int(v) for v in self.np.unique(anchor_mask) if int(v) > 0)
        if not active_obj_ids:
            self.messagebox.showerror(
                "Cutie-lite",
                "No prompts found on current frame. Add at least one left/right click before propagation.",
            )
            return

        self.propagating = True
        self._set_ui_for_propagation(True)
        self.progress_var.set(0.0)
        self.status_var.set(
            f"Propagating {direction} from frame {anchor_idx + 1} with objects {active_obj_ids}..."
        )

        self.propagation_thread = threading.Thread(
            target=self._propagate_worker,
            args=(direction, anchor_idx, anchor_mask.astype(self.np.uint8), active_obj_ids),
            daemon=True,
        )
        self.propagation_thread.start()

    def _propagate_worker(self, direction: str, anchor_idx: int, anchor_mask, active_obj_ids: list[int]) -> None:
        import torch

        try:
            processor = self.InferenceCore(self.cutie, self.cfg)
            if direction == "forward":
                indices = list(range(anchor_idx, self.total_frames))
            else:
                indices = list(range(anchor_idx, -1, -1))

            amp_enabled = bool(getattr(self.cfg, "amp", False) and self.cfg.device == "cuda")
            autocast_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if self.cfg.device == "cuda" else nullcontext()

            with torch.inference_mode(), autocast_ctx:
                first_mask_t = torch.from_numpy(anchor_mask).long().to(self.cfg.device, non_blocking=True)
                total = len(indices)
                for step, frame_idx in enumerate(indices):
                    frame = self.res_man.get_image(frame_idx)
                    image_t = self.image_to_torch(frame, device=self.cfg.device)

                    if step == 0:
                        out_prob = processor.step(
                            image_t,
                            first_mask_t,
                            objects=active_obj_ids,
                            idx_mask=True,
                            end=(step == total - 1),
                            force_permanent=True,
                        )
                    else:
                        out_prob = processor.step(image_t, end=(step == total - 1))

                    out_mask = processor.output_prob_to_mask(out_prob).to(torch.uint8).cpu().numpy()
                    with self.mask_lock:
                        self.masks_by_frame[frame_idx] = out_mask.astype(self.np.uint8)

                    if step % 5 == 0 or step == total - 1:
                        p = float(step + 1) / max(1, total)
                        self.root.after(
                            0,
                            lambda p=p, step=step, total=total, direction=direction: self._on_propagation_progress(
                                direction, step + 1, total, p
                            ),
                        )

            self.root.after(
                0, lambda direction=direction, total=total: self._on_propagation_done(direction, total)
            )
        except Exception as exc:
            self.root.after(0, lambda err=str(exc): self._on_propagation_failed(err))
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _on_propagation_progress(self, direction: str, current: int, total: int, progress: float) -> None:
        self.progress_var.set(progress)
        self.status_var.set(f"Propagating {direction}: {current}/{total}")
        self._refresh_frame()

    def _on_propagation_done(self, direction: str, total: int) -> None:
        self.propagating = False
        self._set_ui_for_propagation(False)
        self.progress_var.set(1.0)
        self.status_var.set(f"Propagation {direction} complete ({total} frames).")
        self._refresh_frame()

    def _on_propagation_failed(self, error: str) -> None:
        self.propagating = False
        self._set_ui_for_propagation(False)
        self.status_var.set("Propagation failed.")
        lower = error.lower()
        if "out of memory" in lower and "cuda" in lower:
            hint = (
                "\n\nCUDA OOM hint:\n"
                "- Close other GPU apps\n"
                "- Retry with smaller internal size (e.g. --max-internal-size 360)\n"
                "- Set EZFRAMES_CUTIE_CHUNK_SIZE=2 and retry"
            )
            self.messagebox.showerror("Cutie-lite propagation failed", f"{error}{hint}")
            return
        self.messagebox.showerror("Cutie-lite propagation failed", error)

    def _export_masks(self) -> None:
        if self.propagating:
            return
        self._stop_playback()
        with self.mask_lock:
            masks = {k: v.copy() for k, v in self.masks_by_frame.items()}

        if not masks:
            self.messagebox.showerror("Cutie-lite", "No masks available to export. Propagate first.")
            return

        active_obj_ids = list(self.active_obj_ids)
        if not active_obj_ids:
            discovered: set[int] = set()
            for mask in masks.values():
                discovered.update(int(v) for v in self.np.unique(mask) if int(v) > 0)
            active_obj_ids = sorted(discovered)
        if not active_obj_ids:
            self.messagebox.showerror("Cutie-lite", "No active object IDs found in masks.")
            return

        if self.output_mask_dir.exists():
            shutil.rmtree(self.output_mask_dir, ignore_errors=True)
        self.output_mask_dir.mkdir(parents=True, exist_ok=True)

        blank = self.np.zeros((self.h, self.w), dtype=self.np.uint8)
        for frame_idx in range(self.total_frames):
            src = masks.get(frame_idx, blank)
            binary = self.np.where(self.np.isin(src, active_obj_ids), 255, 0).astype(self.np.uint8)
            out_path = self.output_mask_dir / f"{frame_idx:07d}.png"
            self.cv2.imwrite(str(out_path), binary)

        self.exported_count = self.total_frames
        self.status_var.set(
            f"Export complete: {self.exported_count} masks to {self.output_mask_dir} (objects {active_obj_ids})"
        )
        self.messagebox.showinfo("Cutie-lite", "Binary mask export complete.")

    def _refresh_frame(self) -> None:
        frame = self.res_man.get_image(self.current_idx)
        mask = self._current_mask()
        vis = self.get_visualization("davis", frame, mask, None, list(range(1, self.num_objects + 1)))

        max_w = max(320, self.image_label.winfo_width() - 4)
        max_h = max(180, self.image_label.winfo_height() - 4)
        src_h, src_w = vis.shape[:2]
        if max_w <= 1 or max_h <= 1:
            max_w, max_h = 1120, 660

        scale = min(max_w / max(1, src_w), max_h / max(1, src_h), 1.0)
        draw_w = max(1, int(src_w * scale))
        draw_h = max(1, int(src_h * scale))
        if draw_w != src_w or draw_h != src_h:
            vis = self.cv2.resize(vis, (draw_w, draw_h), interpolation=self.cv2.INTER_AREA)

        self.display_scale = scale
        self.display_w = draw_w
        self.display_h = draw_h

        image = self.Image.fromarray(vis)
        self.frame_photo = self.ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.frame_photo, text="")

        self.frame_var.set(
            f"Frame {self.current_idx + 1}/{self.total_frames} | FPS {round(self.fps, 2)} | Object {self.current_obj}"
        )

    def _on_close(self) -> None:
        if self.propagating:
            if not self.messagebox.askyesno("Cutie-lite", "Propagation is running. Close anyway?"):
                return
        if self.exported_count == 0:
            if not self.messagebox.askyesno(
                "Cutie-lite",
                "Masks have not been exported. Close anyway and discard this session?",
            ):
                return
        self.closed = True
        self._stop_playback()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> int:
        self.root.mainloop()
        return int(self.exported_count)


def _run_session(cfg, output_mask_dir: Path) -> int:
    session = CutieLiteSession(cfg, output_mask_dir)
    try:
        return session.run()
    finally:
        try:
            session.res_man.cleanup()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = Path(args.video).resolve()
    workspace = Path(args.workspace).resolve()
    output_mask_dir = Path(args.output_mask_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    output_mask_dir.parent.mkdir(parents=True, exist_ok=True)

    paths = AppPaths.default()
    cutie_root = None

    try:
        cutie_root = _resolve_cutie_root(paths)
        _prepare_cutie_import_paths(cutie_root)

        selected_device = _select_device(args.device)
        cutie_weights = _resolve_weight_file(
            "cutie-base-mega.pth", paths=paths, cutie_root=cutie_root, env_var="EZFRAMES_CUTIE_WEIGHTS"
        )
        ritm_weights = _resolve_weight_file(
            "coco_lvis_h18_itermask.pth", paths=paths, cutie_root=cutie_root, env_var="EZFRAMES_RITM_WEIGHTS"
        )
        used_fallback = False

        def _run_on_device(device_name: str) -> int:
            cfg = _build_cfg(
                cutie_root=cutie_root,
                video_path=video,
                workspace=workspace,
                num_objects=int(args.num_objects),
                device=device_name,
                cutie_weights=cutie_weights,
                ritm_weights=ritm_weights,
                mem_every=args.mem_every,
                max_internal_size=args.max_internal_size,
                max_overall_size=args.max_overall_size,
            )
            return _run_session(cfg, output_mask_dir)

        try:
            count = _run_on_device(selected_device)
        except Exception as exc:
            if selected_device == "cuda" and _is_cuda_error(exc):
                log.warning("Cutie-lite CUDA startup failed; retrying on CPU: %s", exc)
                selected_device = "cpu"
                used_fallback = True
                count = _run_on_device(selected_device)
            else:
                raise

        if count <= 0:
            raise RuntimeError("Cutie-lite session ended without exported masks.")

        payload = {
            "ok": True,
            "video": str(video),
            "workspace": str(workspace),
            "mask_dir": str(output_mask_dir),
            "mask_count": count,
            "device": selected_device,
            "device_fallback": "cuda->cpu" if used_fallback else "",
            "cutie_root": str(cutie_root),
        }
        _write_status(args.status_file, payload)
        log.info("Cutie-lite complete: %s", payload)
        return 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "video": str(video),
            "workspace": str(workspace),
            "mask_dir": str(output_mask_dir),
            "cutie_root": str(cutie_root) if cutie_root else "",
        }
        _write_status(args.status_file, payload)
        print(f"Cutie-lite failure: {exc}", file=sys.stderr)
        return 4

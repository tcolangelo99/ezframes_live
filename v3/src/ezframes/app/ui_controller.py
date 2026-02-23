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
import tkinter.font as tkfont
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk
except Exception:  # pragma: no cover - optional runtime dependency
    ctk = None

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
from ezframes.common.entitlements import Entitlement, resolve_entitlement_from_env
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
        self.entitlement: Entitlement = resolve_entitlement_from_env(default_tier="free")
        self._ctk_available = ctk is not None
        if self._ctk_available:
            try:
                ctk.set_appearance_mode("dark")
            except Exception:
                pass
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
        self._brand_logo: ImageTk.PhotoImage | None = None
        self._theme_colors: dict[str, str] = {}
        self._pro_notice_shown = False

        self._build_ui()
        self._apply_entitlement_gates()
        self._warn_if_no_cuda()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _icon_candidates(self) -> list[Path]:
        candidates: list[Path] = [
            self.paths.assets_dir / "icons" / "ezframes_icon.ico",
            self.paths.assets_dir / "icons" / "launcher_icon.ico",
            self.paths.install_root / "icons" / "ezframes_icon.ico",
        ]
        for root in self.paths.source_roots():
            candidates.append(root / "icons" / "ezframes_icon.ico")
            candidates.append(root / "ezframes_icon.ico")
            candidates.append(root / "launcher_icon.ico")
        return candidates

    def _apply_window_icon(self, window: tk.Misc) -> None:
        for candidate in self._icon_candidates():
            try:
                if candidate.exists():
                    window.iconbitmap(str(candidate))
                    return
            except Exception:
                continue

    def _logo_candidates(self) -> list[Path]:
        candidates: list[Path] = [
            self.paths.assets_dir / "icons" / "EzFrames_logo_min.png",
            self.paths.assets_dir / "icons" / "EzFrames_logo.png",
        ]
        for root in self.paths.source_roots():
            candidates.append(root / "icons" / "EzFrames_logo_min.png")
            candidates.append(root / "icons" / "EzFrames_logo.png")
        return candidates

    def _load_brand_logo(self, max_width: int = 220, max_height: int = 80) -> ImageTk.PhotoImage | None:
        for candidate in self._logo_candidates():
            if not candidate.exists():
                continue
            try:
                image = Image.open(candidate)
                image.thumbnail((max_width, max_height))
                self._brand_logo = ImageTk.PhotoImage(image)
                return self._brand_logo
            except Exception:
                continue
        return None

    def _is_pro(self) -> bool:
        return bool(self.entitlement.is_pro)

    def _entitlement_caption(self) -> str:
        if self._is_pro():
            return "Pro tier unlocked"
        return "Free tier: Absdiff + ORB (advanced features are Pro)"

    def _show_pro_required(self, feature_name: str) -> None:
        msg = (
            f"{feature_name} is available on EzFrames Pro.\n\n"
            "Free tier includes full tuning for Absdiff + ORB.\n"
            "Sign in with an active subscription to unlock this feature."
        )
        if not self._pro_notice_shown:
            self._pro_notice_shown = True
        messagebox.showinfo("EzFrames Pro Feature", msg)

    def _require_pro_feature(self, feature_name: str) -> bool:
        if self._is_pro():
            return True
        self._show_pro_required(feature_name)
        return False

    def _set_widget_state_generic(self, widget, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        tk_state = tk.NORMAL if enabled else tk.DISABLED
        for candidate_state in (state, tk_state):
            try:
                widget.configure(state=candidate_state)
                return
            except Exception:
                continue

    def _apply_entitlement_gates(self) -> None:
        self.tier_var.set(self._entitlement_caption())
        if self._is_pro():
            self.var_status.set("Ready.")
            return

        # Free-tier hard constraints: full tuning remains available for Absdiff + ORB.
        self.var_motion_method.set("Absdiff")
        self.var_detector.set("ORB")
        self.var_use_masks.set(False)
        self.var_output_masked.set(False)
        self.var_output_images.set(False)
        self.var_debug.set(False)
        self.var_interp_factor.set("2")

        self._set_widget_state_generic(self.motion_combo, False)
        self._set_widget_state_generic(self.detector_combo, False)
        self._set_widget_state_generic(self.use_masks_check, False)
        self._set_widget_state_generic(self.output_masked_check, False)
        self._set_widget_state_generic(self.output_images_check, False)
        self._set_widget_state_generic(self.debug_check, False)
        self._set_widget_state_generic(self.cutie_objects_spin, False)
        self._set_widget_state_generic(self.interp_combo, False)
        self._set_widget_state_generic(self.btn_track_cutie, False)
        self._set_widget_state_generic(self.btn_interpolate, False)

        self.var_status.set("Free mode active: Absdiff + ORB. Sign in for Cutie, RIFE, and advanced outputs.")

    @staticmethod
    def _pick_font(candidates: list[str], fallback: str = "Segoe UI") -> str:
        try:
            available = {name.lower() for name in tkfont.families()}
        except Exception:
            available = set()
        if not available:
            return candidates[0] if candidates else fallback
        for candidate in candidates:
            if candidate.lower() in available:
                return candidate
        return fallback

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        body_font = self._pick_font(["Segoe UI", "Calibri", "Arial"])
        accent_font = self._pick_font(["WWE Raw", "Segoe UI Semibold", "Segoe UI"], fallback=body_font)
        self._theme_colors = {
            "bg": "#1A1A1A",
            "panel": "#242424",
            "panel_alt": "#313131",
            "text": "#FFFFFF",
            "muted": "#BCC7D0",
            "accent": "#00F0FF",
            "accent_hover": "#49F3FF",
            "primary": "#175482",
            "primary_hover": "#2A6D9E",
            "secondary": "#394955",
            "secondary_hover": "#6D8B93",
            "video_bg": "#101010",
            "danger": "#D16A74",
        }
        c = self._theme_colors
        self.root.configure(bg=c["bg"])

        style.configure(".", background=c["bg"], foreground=c["text"], font=(body_font, 10))
        style.configure("TFrame", background=c["bg"])
        style.configure("Card.TFrame", background=c["panel"])
        style.configure("TLabel", background=c["bg"], foreground=c["text"], font=(body_font, 10))
        style.configure("Card.TLabel", background=c["panel"], foreground=c["text"])
        style.configure("Muted.TLabel", background=c["bg"], foreground=c["muted"], font=(body_font, 10))
        style.configure("TLabelframe", background=c["panel"], bordercolor=c["panel_alt"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=c["panel"], foreground=c["accent"], font=(accent_font, 10, "bold"))
        style.configure("TCheckbutton", background=c["panel"], foreground=c["text"])
        style.map("TCheckbutton", background=[("active", c["panel"])], foreground=[("disabled", c["muted"])])

        style.configure(
            "TButton",
            background=c["secondary"],
            foreground=c["text"],
            borderwidth=1,
            focusthickness=0,
            padding=(12, 7),
            font=(body_font, 11, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", c["secondary_hover"]), ("disabled", c["panel_alt"])],
            foreground=[("disabled", c["muted"])],
        )
        style.configure(
            "Secondary.TButton",
            background=c["secondary"],
            foreground=c["text"],
            borderwidth=1,
            focusthickness=0,
            padding=(12, 7),
            font=(body_font, 11, "bold"),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", c["secondary_hover"]), ("disabled", c["panel_alt"])],
            foreground=[("disabled", c["muted"])],
        )
        style.configure(
            "Primary.TButton",
            background=c["primary"],
            foreground=c["text"],
            borderwidth=1,
            focusthickness=0,
            padding=(16, 8),
            font=(body_font, 12, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", c["secondary_hover"]), ("disabled", c["panel_alt"])],
            foreground=[("disabled", c["muted"])],
        )
        style.configure(
            "Player.TButton",
            background=c["panel_alt"],
            foreground=c["text"],
            borderwidth=1,
            focusthickness=0,
            padding=(12, 6),
            font=(body_font, 11, "bold"),
        )
        style.map(
            "Player.TButton",
            background=[("active", c["secondary_hover"]), ("disabled", c["panel_alt"])],
            foreground=[("disabled", c["muted"])],
        )

        style.configure(
            "TEntry",
            fieldbackground=c["panel_alt"],
            foreground=c["text"],
        )
        try:
            style.configure(
                "Dark.TSpinbox",
                fieldbackground=c["panel_alt"],
                foreground=c["text"],
                arrowsize=14,
            )
        except tk.TclError:
            style.configure(
                "Dark.TSpinbox",
                fieldbackground=c["panel_alt"],
                foreground=c["text"],
            )
        try:
            style.configure(
                "Dark.TCombobox",
                fieldbackground=c["panel_alt"],
                foreground=c["text"],
                arrowsize=14,
                bordercolor=c["secondary"],
            )
        except tk.TclError:
            style.configure(
                "Dark.TCombobox",
                fieldbackground=c["panel_alt"],
                foreground=c["text"],
            )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", c["panel_alt"])],
            foreground=[("readonly", c["text"])],
            selectbackground=[("readonly", c["panel_alt"])],
            selectforeground=[("readonly", c["text"])],
        )
        style.map(
            "Dark.TSpinbox",
            fieldbackground=[("readonly", c["panel_alt"]), ("!disabled", c["panel_alt"])],
            foreground=[("readonly", c["text"]), ("!disabled", c["text"])],
        )
        style.configure(
            "Dark.Horizontal.TScale",
            background=c["panel"],
            troughcolor=c["panel_alt"],
            bordercolor=c["panel_alt"],
            lightcolor=c["panel_alt"],
            darkcolor=c["panel_alt"],
        )
        style.configure("Horizontal.TScale", background=c["panel"], troughcolor=c["panel_alt"])
        try:
            style.configure(
                "TProgressbar",
                background=c["accent"],
                darkcolor=c["accent"],
                lightcolor=c["accent"],
                troughcolor=c["panel_alt"],
                bordercolor=c["panel_alt"],
            )
        except tk.TclError:
            style.configure(
                "TProgressbar",
                background=c["accent"],
                troughcolor=c["panel_alt"],
            )
        # Dark dropdown/list colors on Windows ttk popdowns.
        self.root.option_add("*TCombobox*Listbox*Background", c["panel_alt"])
        self.root.option_add("*TCombobox*Listbox*Foreground", c["text"])
        self.root.option_add("*TCombobox*Listbox*selectBackground", c["secondary"])
        self.root.option_add("*TCombobox*Listbox*selectForeground", c["text"])
        self.root.option_add("*TSpinbox*Background", c["panel_alt"])
        self.root.option_add("*TSpinbox*Foreground", c["text"])

    def _build_ui(self) -> None:
        self.root.title("EzFrames")
        self.root.geometry("1280x860")
        self.root.minsize(1120, 760)
        self._configure_style()
        self._apply_window_icon(self.root)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(2, 10))
        header.columnconfigure(1, weight=1)

        logo = self._load_brand_logo()
        if logo is not None:
            logo_label = tk.Label(header, image=logo, bg=self._theme_colors["bg"])
            logo_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))

        tk.Label(
            header,
            text="EzFrames",
            font=(self._pick_font(["WWE Raw", "Segoe UI Semibold", "Segoe UI"]), 28),
            bg=self._theme_colors["bg"],
            fg=self._theme_colors["text"],
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            header,
            text="Motion-cleaning for anime workflows | Cutie masks | RIFE interpolation",
            style="Muted.TLabel",
        ).grid(row=1, column=1, sticky="w", pady=(1, 0))
        self.tier_var = tk.StringVar(value=self._entitlement_caption())
        ttk.Label(
            header,
            textvariable=self.tier_var,
            style="Muted.TLabel",
        ).grid(row=2, column=1, sticky="w", pady=(1, 0))

        self.var_video = tk.StringVar()
        self.var_output = tk.StringVar(value=str(self.paths.workspace_dir))
        self.var_status = tk.StringVar(value="Ready.")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_frame_label = tk.StringVar(value="No video loaded")
        self.var_min_area_label = tk.StringVar(value="Min Contour Area: 100 px")
        self.var_camera_label = tk.StringVar(value="Cam Motion Sensitivity: 25%")
        self.var_area_ratio_label = tk.StringVar(value="Area Ratio: 0.50")
        self.var_sift_scale_label = tk.StringVar(value="SIFT Downscale: 50%")

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
        self.var_interp_factor = tk.StringVar(value="2")

        self._add_path_row(main, 1, "Video", self.var_video, self._browse_video)
        self._add_path_row(main, 2, "Output Dir", self.var_output, self._browse_output)

        opts = ttk.LabelFrame(main, text="Processing Options", padding=8)
        opts.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 6))

        self.use_masks_check = ttk.Checkbutton(opts, text="Use Cutie Masks", variable=self.var_use_masks, command=self._sync_mask_state)
        self.use_masks_check.grid(row=0, column=0, sticky="w", padx=3)
        self.output_masked_check = ttk.Checkbutton(opts, text="Output Masked Video (Alpha)", variable=self.var_output_masked)
        self.output_masked_check.grid(row=0, column=1, sticky="w", padx=3)
        self.output_images_check = ttk.Checkbutton(opts, text="Output Image Sequence", variable=self.var_output_images)
        self.output_images_check.grid(
            row=0, column=2, sticky="w", padx=3
        )
        self.camera_check = ttk.Checkbutton(opts, text="Camera Motion Filter", variable=self.var_camera, command=self._sync_camera_state)
        self.camera_check.grid(
            row=0, column=3, sticky="w", padx=3
        )
        self.debug_check = ttk.Checkbutton(opts, text="Debug Output", variable=self.var_debug)
        self.debug_check.grid(row=0, column=4, sticky="w", padx=3)

        ttk.Label(opts, text="Motion Method").grid(row=1, column=0, sticky="w")
        self.motion_combo = self._make_dark_option_menu(
            opts,
            variable=self.var_motion_method,
            values=["Absdiff", "Optical Flow"],
            width=12,
            command=lambda _v: self._on_motion_method_change(),
        )
        self.motion_combo.grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="Feature Detector").grid(row=1, column=2, sticky="w")
        self.detector_combo = self._make_dark_option_menu(
            opts,
            variable=self.var_detector,
            values=["ORB", "SIFT"],
            width=8,
            command=lambda _v: self._sync_sift_state(),
        )
        self.detector_combo.grid(row=1, column=3, sticky="w")

        ttk.Label(opts, textvariable=self.var_sift_scale_label).grid(row=1, column=4, sticky="w")
        self.sift_slider = self._make_dark_scale(
            opts,
            from_value=0.3,
            to_value=1.0,
            variable=self.var_sift_scale,
            length=140,
            resolution=0.05,
            command=lambda value: self._on_sift_scale_change(float(value)),
        )
        self.sift_slider.grid(row=1, column=5, sticky="w")

        ttk.Label(opts, textvariable=self.var_min_area_label).grid(row=2, column=0, sticky="w")
        self.min_area_slider = self._make_dark_scale(
            opts,
            from_value=0.0,
            to_value=4.0,
            variable=self.var_min_area_log,
            length=170,
            command=lambda value: self._on_min_area_change(float(value)),
        )
        self.min_area_slider.grid(row=2, column=1, sticky="w")
        ttk.Label(opts, textvariable=self.var_area_ratio_label).grid(row=2, column=2, sticky="w")
        self.area_ratio_slider = self._make_dark_scale(
            opts,
            from_value=0.1,
            to_value=1.0,
            variable=self.var_area_ratio,
            length=140,
            resolution=0.01,
            command=lambda value: self._on_area_ratio_change(float(value)),
        )
        self.area_ratio_slider.grid(row=2, column=3, sticky="w")
        ttk.Label(opts, textvariable=self.var_camera_label).grid(row=2, column=4, sticky="w")
        self.camera_slider = self._make_dark_scale(
            opts,
            from_value=1.0,
            to_value=0.1,
            variable=self.var_camera_threshold,
            length=140,
            resolution=0.01,
            command=lambda value: self._on_camera_threshold_change(float(value)),
        )
        self.camera_slider.grid(row=2, column=5, sticky="w")

        ttk.Label(opts, text="ProRes").grid(row=3, column=0, sticky="w")
        self.prores_combo = self._make_dark_option_menu(
            opts,
            variable=self.var_prores,
            values=["proxy", "lt", "standard", "hq", "4444", "4444xq"],
            width=10,
        )
        self.prores_combo.grid(row=3, column=1, sticky="w")
        ttk.Label(opts, text="Cutie Objects").grid(row=3, column=2, sticky="w")
        self.cutie_objects_spin = self._make_dark_spinbox(opts, self.var_cutie_objects, from_value=1, to_value=8, width=4)
        self.cutie_objects_spin.grid(row=3, column=3, sticky="w")
        ttk.Label(opts, text="Interp Factor").grid(row=3, column=4, sticky="w")
        self.interp_combo = self._make_dark_option_menu(
            opts,
            variable=self.var_interp_factor,
            values=[2, 4, 8, 16],
            width=6,
        )
        self.interp_combo.grid(row=3, column=5, sticky="w")

        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        self.btn_track_cutie = self._make_action_button(buttons, "Track with Cutie", self._run_cutie_thread, variant="secondary")
        self.btn_track_cutie.pack(side=tk.LEFT, padx=4)
        self.btn_process = self._make_action_button(buttons, "Process Video", self._process_video_thread, variant="primary")
        self.btn_process.pack(side=tk.LEFT, padx=4)
        self.btn_interpolate = self._make_action_button(buttons, "Interpolate Last Output", self._interpolate_thread, variant="secondary")
        self.btn_interpolate.pack(side=tk.LEFT, padx=4)

        preview = ttk.LabelFrame(main, text="Preview", padding=6)
        preview.grid(row=5, column=0, columnspan=4, sticky="nsew", pady=(6, 6))
        preview.grid_propagate(False)
        self.preview_container = preview
        preview_surface = tk.Frame(
            preview,
            bg=self._theme_colors["video_bg"],
            highlightbackground=self._theme_colors["panel_alt"],
            highlightthickness=1,
        )
        preview_surface.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        self.preview_label = tk.Label(
            preview_surface,
            text="No preview",
            bg=self._theme_colors["video_bg"],
            fg=self._theme_colors["muted"],
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        pctrl = ttk.Frame(preview)
        pctrl.pack(fill=tk.X)
        self._make_action_button(pctrl, "Previous", self._preview_prev, variant="player").pack(side=tk.LEFT, padx=4)
        self.play_btn = self._make_action_button(pctrl, "Play", self._preview_toggle, variant="player")
        self.play_btn.pack(side=tk.LEFT, padx=4)
        self._make_action_button(pctrl, "Next", self._preview_next, variant="player").pack(side=tk.LEFT, padx=4)
        ttk.Label(pctrl, textvariable=self.var_frame_label).pack(side=tk.LEFT, padx=10)

        ttk.Progressbar(main, variable=self.var_progress, maximum=1.0).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        ttk.Label(main, textvariable=self.var_status, anchor="w", style="Muted.TLabel").grid(row=7, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        for col in range(4):
            main.columnconfigure(col, weight=1)
        main.rowconfigure(5, weight=1)

        self.var_output.trace_add("write", self._on_output_dir_changed)
        self._on_motion_method_change()
        self._on_area_ratio_change(float(self.var_area_ratio.get()))
        self._on_sift_scale_change(float(self.var_sift_scale.get()))
        self._on_camera_threshold_change(float(self.var_camera_threshold.get()))
        self._sync_mask_state()
        self._sync_camera_state()
        self._sync_sift_state()

    def _add_path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, browse_cmd) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        self._make_action_button(parent, "Browse", browse_cmd, variant="secondary", padx=10, pady=5).grid(
            row=row, column=3, sticky="ew"
        )

    def _make_action_button(
        self,
        parent,
        text: str,
        command,
        variant: str = "secondary",
        padx: int = 12,
        pady: int = 7,
    ):
        c = self._theme_colors
        bg = c["secondary"]
        font_size = 11
        corner_radius = 18
        if variant == "primary":
            bg = c["primary"]
            font_size = 12
            corner_radius = 20
        elif variant == "player":
            bg = c["panel_alt"]
            font_size = 11
            corner_radius = 14
        if self._ctk_available and ctk is not None:
            btn = ctk.CTkButton(
                parent,
                text=text,
                command=command,
                fg_color=bg,
                hover_color=c["secondary_hover"],
                text_color=c["text"],
                corner_radius=corner_radius,
                border_width=0,
                height=max(32, 18 + pady * 2),
                font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), font_size, "bold"),
            )
            return btn
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=c["text"],
            activebackground=c["secondary_hover"],
            activeforeground=c["text"],
            disabledforeground=c["muted"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=c["panel_alt"],
            highlightcolor=c["accent"],
            cursor="hand2",
            padx=padx,
            pady=pady,
            font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), font_size, "bold"),
        )
        return btn

    def _make_dark_option_menu(
        self,
        parent,
        variable: tk.Variable,
        values: list,
        width: int = 10,
        command=None,
    ):
        c = self._theme_colors
        if self._ctk_available and ctk is not None:
            values_str = [str(v) for v in values]
            option = ctk.CTkOptionMenu(
                parent,
                variable=variable,
                values=values_str,
                command=command,
                fg_color=c["secondary"],
                button_color=c["secondary"],
                button_hover_color=c["secondary_hover"],
                text_color=c["text"],
                dropdown_fg_color=c["panel_alt"],
                dropdown_text_color=c["text"],
                dropdown_hover_color=c["secondary_hover"],
                corner_radius=8,
                width=max(84, int(width * 10)),
                font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), 10, "bold"),
                dropdown_font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), 10),
            )
            return option
        option = tk.OptionMenu(parent, variable, *values, command=command)
        option.configure(
            bg=c["panel_alt"],
            fg=c["text"],
            activebackground=c["secondary_hover"],
            activeforeground=c["text"],
            highlightthickness=1,
            highlightbackground=c["panel_alt"],
            highlightcolor=c["accent"],
            bd=0,
            relief=tk.FLAT,
            width=width,
            font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), 10, "bold"),
            cursor="hand2",
        )
        menu = option["menu"]
        menu.configure(
            bg=c["panel_alt"],
            fg=c["text"],
            activebackground=c["secondary_hover"],
            activeforeground=c["text"],
            bd=0,
            relief=tk.FLAT,
            font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), 10),
        )
        return option

    def _make_dark_spinbox(
        self,
        parent,
        variable: tk.Variable,
        from_value: int,
        to_value: int,
        width: int = 4,
    ) -> tk.Spinbox:
        c = self._theme_colors
        spin = tk.Spinbox(
            parent,
            from_=from_value,
            to=to_value,
            textvariable=variable,
            width=width,
            bg=c["panel_alt"],
            fg=c["text"],
            buttonbackground=c["secondary"],
            activebackground=c["secondary_hover"],
            disabledbackground=c["panel_alt"],
            disabledforeground=c["muted"],
            insertbackground=c["text"],
            readonlybackground=c["panel_alt"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=c["panel_alt"],
            highlightcolor=c["accent"],
            font=(self._pick_font(["Segoe UI", "Calibri", "Arial"]), 10, "bold"),
        )
        return spin

    def _make_dark_scale(
        self,
        parent,
        from_value: float,
        to_value: float,
        variable: tk.Variable,
        length: int = 140,
        command=None,
        resolution: float = 0.01,
    ):
        c = self._theme_colors
        if self._ctk_available and ctk is not None:
            steps = 0
            try:
                if resolution > 0:
                    steps = int(round(abs(float(to_value) - float(from_value)) / float(resolution)))
            except Exception:
                steps = 0
            slider_kwargs = {}
            if steps > 0:
                slider_kwargs["number_of_steps"] = steps
            scale = ctk.CTkSlider(
                parent,
                from_=from_value,
                to=to_value,
                variable=variable,
                command=command,
                fg_color=c["panel_alt"],
                progress_color=c["accent"],
                button_color=c["secondary"],
                button_hover_color=c["secondary_hover"],
                width=length,
                height=18,
                **slider_kwargs,
            )
            return scale
        scale = tk.Scale(
            parent,
            from_=from_value,
            to=to_value,
            variable=variable,
            orient=tk.HORIZONTAL,
            length=length,
            showvalue=0,
            sliderlength=18,
            resolution=resolution,
            command=command,
            bg=c["panel"],
            fg=c["text"],
            troughcolor=c["panel_alt"],
            activebackground=c["secondary"],
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
        )
        return scale

    def _set_scale_state(self, scale, enabled: bool) -> None:
        c = self._theme_colors
        if self._ctk_available and ctk is not None and isinstance(scale, ctk.CTkSlider):
            if enabled:
                scale.configure(
                    state="normal",
                    fg_color=c["panel_alt"],
                    progress_color=c["accent"],
                    button_color=c["secondary"],
                    button_hover_color=c["secondary_hover"],
                )
            else:
                scale.configure(
                    state="disabled",
                    fg_color="#2A2A2A",
                    progress_color="#2A2A2A",
                    button_color="#555555",
                    button_hover_color="#555555",
                )
            return
        if enabled:
            scale.configure(
                state=tk.NORMAL,
                troughcolor=c["panel_alt"],
                activebackground=c["secondary"],
                bg=c["panel"],
            )
        else:
            scale.configure(
                state=tk.DISABLED,
                troughcolor="#2A2A2A",
                activebackground="#2A2A2A",
                bg=c["panel"],
            )

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

    def _on_area_ratio_change(self, value: float) -> None:
        self.var_area_ratio_label.set(f"Area Ratio: {float(value):.2f}")

    def _on_sift_scale_change(self, value: float) -> None:
        pct = int(round(float(value) * 100))
        self.var_sift_scale_label.set(f"SIFT Downscale: {pct}%")

    def _on_motion_method_change(self) -> None:
        method = self.var_motion_method.get()
        if not self._is_pro():
            if method != "Absdiff":
                self.var_motion_method.set("Absdiff")
            method = "Absdiff"
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
        if not self._is_pro():
            self.var_use_masks.set(False)
            self.var_output_masked.set(False)
            self.use_masks_check.configure(state="disabled")
            self.output_masked_check.configure(state="disabled")
            return
        available = self._refresh_mask_availability()
        if self.var_use_masks.get() and available:
            self.output_masked_check.configure(state="normal")
        else:
            self.output_masked_check.configure(state="disabled")
            self.var_output_masked.set(False)

    def _sync_camera_state(self) -> None:
        if not self._is_pro():
            self.var_detector.set("ORB")
            self.detector_combo.configure(state="disabled")
            self._set_scale_state(self.camera_slider, enabled=bool(self.var_camera.get()))
            self._sync_sift_state()
            return
        if self.var_camera.get():
            self.detector_combo.configure(state="normal")
            self._set_scale_state(self.camera_slider, enabled=True)
        else:
            self.detector_combo.configure(state="disabled")
            self._set_scale_state(self.camera_slider, enabled=False)
        self._sync_sift_state()

    def _sync_sift_state(self) -> None:
        if not self._is_pro():
            self.var_detector.set("ORB")
            self._set_scale_state(self.sift_slider, enabled=False)
            return
        if self.var_camera.get() and self.var_detector.get() == "SIFT":
            self._set_scale_state(self.sift_slider, enabled=True)
        else:
            self._set_scale_state(self.sift_slider, enabled=False)

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

    @staticmethod
    def _subprocess_flags() -> int:
        if os.name == "nt":
            return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return 0

    def _cuda_support(self) -> tuple[bool, bool]:
        cv2_cuda = False
        torch_cuda = False

        try:
            cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception as exc:
            log.info("OpenCV CUDA probe failed: %s", exc)
            cv2_cuda = False

        try:
            import torch

            torch_cuda = bool(torch.cuda.is_available())
            if not torch_cuda:
                log.info("PyTorch CUDA not available. torch=%s cuda=%s", getattr(torch, "__version__", "?"), getattr(torch.version, "cuda", "?"))
        except Exception as exc:
            log.warning("PyTorch CUDA probe failed: %s", exc)
            torch_cuda = False

        return cv2_cuda, torch_cuda

    def _warn_if_no_cuda(self) -> None:
        cv2_cuda, torch_cuda = self._cuda_support()
        if not cv2_cuda and not torch_cuda:
            nvidia_detail = ""
            try:
                info = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                    creationflags=self._subprocess_flags(),
                )
                if info.returncode == 0 and info.stdout.strip():
                    nvidia_detail = f"\nDetected GPU: {info.stdout.strip().splitlines()[0]}"
            except Exception as exc:
                log.info("nvidia-smi probe failed: %s", exc)

            messagebox.showwarning(
                "CUDA Not Detected",
                "CUDA acceleration was not detected in this runtime. CPU fallback will be used.\n"
                "A runtime dependency repair/update may be required."
                + nvidia_detail,
            )
            return
        if torch_cuda and not cv2_cuda:
            self.var_status.set(
                "CUDA is available for PyTorch features (Cutie/RIFE), but OpenCV CUDA is unavailable. "
                "Optical-flow path will use CPU."
            )

    def _run_cutie_thread(self) -> None:
        if not self._require_pro_feature("Cutie tracking"):
            return
        threading.Thread(target=self._run_cutie, daemon=True).start()

    @staticmethod
    def _python_can_import_module(python_exe: str, module_name: str, env: dict[str, str] | None = None) -> bool:
        try:
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
            completed = subprocess.run(
                [python_exe, "-c", f"import {module_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                shell=False,
                env=env,
                creationflags=creationflags,
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

        fallback_python: str | None = None
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            c_str = str(candidate)
            if c_str in seen:
                continue
            seen.add(c_str)

            # Native cutie-lite path requires these, not Qt.
            if self._python_can_import_modules(c_str, ["torch", "cv2", "omegaconf", "scipy", "tqdm"], env=env):
                has_ctk = self._python_can_import_module(c_str, "customtkinter", env=env)
                if has_ctk:
                    self._cutie_python = c_str
                    return c_str
                if fallback_python is None:
                    fallback_python = c_str

        if fallback_python is not None:
            self._cutie_python = fallback_python
            return fallback_python

        raise RuntimeError(
            "No Python interpreter with required Cutie-lite dependencies was found "
            "(torch, cv2, omegaconf, scipy, tqdm). "
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
        has_ctk = self._python_can_import_module(cutie_python, "customtkinter")
        log.info("Cutie interpreter selected: %s (customtkinter=%s)", cutie_python, has_ctk)

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
        self.root.after(
            0,
            lambda vw=str(video_workspace), ctk_flag=has_ctk: self.var_status.set(
                f"Launching Cutie-lite ({'CTk' if ctk_flag else 'fallback'} UI, {vw})..."
            ),
        )
        completed = subprocess.run(
            cmd,
            shell=False,
            env=env,
            capture_output=True,
            text=True,
            creationflags=self._subprocess_flags(),
        )
        rc = int(completed.returncode)
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
            stderr_text = (completed.stderr or "").strip()
            if stderr_text:
                log.error("Cutie stderr:\n%s", stderr_text)
                if "No module named" in stderr_text:
                    detail = f"{detail}\n\n{stderr_text.splitlines()[-1]}"
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
        user_output_root = Path(out_dir)
        user_output_root.mkdir(parents=True, exist_ok=True)

        method = self.var_motion_method.get()
        use_masks = bool(self.var_use_masks.get())
        alpha_masked = bool(self.var_output_masked.get()) and use_masks
        output_images = bool(self.var_output_images.get())
        camera_filter = bool(self.var_camera.get())
        debug = bool(self.var_debug.get())
        detector = self.var_detector.get()

        if not self._is_pro():
            method = "Absdiff"
            detector = "ORB"
            use_masks = False
            alpha_masked = False
            output_images = False
            debug = False
            self.var_motion_method.set(method)
            self.var_detector.set(detector)
            self.var_use_masks.set(False)
            self.var_output_masked.set(False)
            self.var_output_images.set(False)
            self.var_debug.set(False)

        opts = MotionOptions(
            min_area=self._min_area_pixels(self.var_min_area_log.get()),
            method=method,
            area_ratio_threshold=float(self.var_area_ratio.get()),
            use_cuda=self._cuda_support()[0],
        )

        sift_scale = float(self.var_sift_scale.get())
        camera_thr = float(self.var_camera_threshold.get())
        prores = self.var_prores.get()

        base = Path(video).stem
        processing_output_dir = user_output_root / "ezframes_output"
        processing_output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = mask_dir_for_video(workspace, video)
        out_video = str(processing_output_dir / f"{base}.mov")
        out_images = processing_output_dir / f"{base}_frames"
        out_homography = processing_output_dir / f"{base}_homography"
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
            self.root.after(
                0,
                lambda d=str(processing_output_dir), t=total: self.var_status.set(
                    f"Processing {t} frames... Output: {d}"
                ),
            )
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
        if not self._require_pro_feature("RIFE interpolation"):
            return
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
    auth_code = enforce_authorized_launch(interactive_error=True, consume_ticket=True)
    if auth_code != 0:
        return auth_code

    paths = AppPaths.default()
    paths.ensure_layout()
    configure_logging(paths.logs_dir)

    def _log_uncaught(exc_type, exc_value, exc_tb) -> None:
        log.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _log_uncaught
    if hasattr(threading, "excepthook"):
        def _thread_excepthook(args) -> None:  # type: ignore[no-redef]
            log.exception(
                "Unhandled thread exception in %s",
                getattr(args.thread, "name", "thread"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        threading.excepthook = _thread_excepthook  # type: ignore[attr-defined]

    root = tk.Tk()
    EzFramesUI(root, paths)
    root.mainloop()
    return 0


from __future__ import annotations

import importlib.util
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ezframes.common.config import AppPaths

log = logging.getLogger(__name__)


@dataclass
class InterpolationRequest:
    input_path: str
    output_path: str
    sf: int
    prores_quality: int


class RifeInterpolator:
    def __init__(self, paths: AppPaths | None = None):
        self.paths = paths or AppPaths.default()
        self._impl = None

    @staticmethod
    def _is_cuda_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "cuda",
            "cudnn",
            "nvrtc",
            "out of memory",
            "device-side assert",
            "driver",
        )
        return any(marker in text for marker in markers)

    def _reset_impl(self) -> None:
        self._impl = None

    @staticmethod
    def _load_motion_interpolator_from_file(path: Path):
        module_name = "_ezframes_motion_interpolation"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "MotionInterpolator"):
            raise RuntimeError(f"MotionInterpolator missing in {path}")
        return module.MotionInterpolator

    def _resolve_motion_interpolator_class(self):
        try:
            from motion_interpolation import MotionInterpolator

            return MotionInterpolator
        except Exception:
            pass

        search_roots = [*self.paths.source_roots(), self.paths.app_dir]
        for root in search_roots:
            candidate = root / "motion_interpolation.py"
            if candidate.exists():
                return self._load_motion_interpolator_from_file(candidate)
        raise RuntimeError("Could not import MotionInterpolator from installed package or source roots.")

    def _resolve_model_dir(self) -> Path:
        env_model = os.environ.get("EZFRAMES_MODEL_DIR", "").strip()
        if env_model:
            p = Path(env_model)
            if (p / "flownet.pkl").exists():
                return p

        env_model_path = os.environ.get("EZFRAMES_FLOWNET_PATH", "").strip()
        if env_model_path:
            p = Path(env_model_path)
            if p.exists():
                return p.parent

        if (self.paths.models_dir / "flownet.pkl").exists():
            return self.paths.models_dir

        for root in self.paths.source_roots():
            if (root / "flownet.pkl").exists():
                return root
            if (root / "models" / "flownet.pkl").exists():
                return root / "models"

        raise RuntimeError("Could not locate flownet.pkl. Set EZFRAMES_MODEL_DIR or EZFRAMES_FLOWNET_PATH.")

    @staticmethod
    @contextmanager
    def _temp_cwd(path: Path):
        prev = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev)

    def _ensure_loaded(self) -> None:
        if self._impl is not None:
            return
        try:
            MotionInterpolator = self._resolve_motion_interpolator_class()
        except Exception as exc:
            raise RuntimeError(f"Could not import MotionInterpolator: {exc}") from exc
        self._impl = MotionInterpolator()
        model_dir = self._resolve_model_dir()
        with self._temp_cwd(model_dir):
            self._impl.load_model()

    def interpolate(self, req: InterpolationRequest, progress_callback=None) -> str:
        self._ensure_loaded()
        assert self._impl is not None
        model_dir = self._resolve_model_dir()

        try:
            with self._temp_cwd(model_dir):
                self._impl.interpolate_video(
                    req.input_path,
                    req.output_path,
                    req.sf,
                    req.prores_quality,
                    progress_callback=progress_callback,
                )
        except Exception as exc:
            requested = os.environ.get("EZFRAMES_RIFE_DEVICE", "auto").strip().lower()
            if requested == "cpu" or not self._is_cuda_error(exc):
                raise
            log.warning("RIFE CUDA path failed; retrying on CPU fallback: %s", exc)
            os.environ["EZFRAMES_RIFE_DEVICE"] = "cpu"
            self._reset_impl()
            self._ensure_loaded()
            assert self._impl is not None
            with self._temp_cwd(model_dir):
                self._impl.interpolate_video(
                    req.input_path,
                    req.output_path,
                    req.sf,
                    req.prores_quality,
                    progress_callback=progress_callback,
                )
        return req.output_path

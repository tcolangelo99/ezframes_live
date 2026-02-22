from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


log = logging.getLogger(__name__)


@dataclass
class MotionOptions:
    min_area: float = 100.0
    method: str = "Absdiff"
    area_ratio_threshold: float = 0.5
    use_cuda: bool = True


@dataclass
class CameraMotionResult:
    is_camera_motion: bool
    homography: np.ndarray | None
    good_matches: list
    kp1: list
    kp2: list
    vis_frame1: np.ndarray
    vis_frame2: np.ndarray


def _opencv_cuda_available() -> bool:
    try:
        return bool(hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:
        return False


def _apply_mask(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return gray
    return np.where(mask > 0, gray, 0).astype(np.uint8)


def detect_camera_motion(
    frame1: np.ndarray,
    frame2: np.ndarray,
    mask: np.ndarray | None = None,
    feature_detector: Literal["ORB", "SIFT"] = "ORB",
    sift_scale: float = 0.5,
    match_threshold: float = 0.75,
    inlier_ratio_threshold: float = 0.75,
) -> CameraMotionResult:
    if feature_detector == "SIFT":
        frame1_proc = resize_frame(frame1, sift_scale)
        frame2_proc = resize_frame(frame2, sift_scale)
        if mask is not None:
            mask_proc = cv2.resize(mask.astype(np.uint8), (frame1_proc.shape[1], frame1_proc.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_proc = None
    else:
        frame1_proc = frame1
        frame2_proc = frame2
        mask_proc = mask

    gray1 = cv2.cvtColor(frame1_proc, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_proc, cv2.COLOR_BGR2GRAY)
    gray1 = _apply_mask(gray1, mask_proc)
    gray2 = _apply_mask(gray2, mask_proc)

    if feature_detector == "SIFT":
        detector = cv2.SIFT_create(nfeatures=500)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
    else:
        detector = cv2.ORB_create(nfeatures=500)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return CameraMotionResult(False, None, [], kp1 or [], kp2 or [], frame1, frame2)

    if feature_detector == "SIFT":
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < match_threshold * n.distance:
                good_matches.append(m)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.match(des1, des2)
        if len(matches) < 8:
            return CameraMotionResult(False, None, [], kp1, kp2, frame1, frame2)
        good_matches = sorted(matches, key=lambda m: m.distance)[: int(len(matches) * match_threshold)]

    if len(good_matches) < 4:
        return CameraMotionResult(False, None, good_matches, kp1, kp2, frame1, frame2)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if inlier_mask is None:
        return CameraMotionResult(False, None, good_matches, kp1, kp2, frame1, frame2)

    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / max(1, len(good_matches))
    is_camera = ratio >= inlier_ratio_threshold

    if feature_detector == "SIFT" and homography is not None:
        # Scale homography back up to original resolution for visualization.
        s = float(sift_scale)
        if s > 0:
            scale_m = np.array([[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]])
            homography = scale_m @ homography @ np.linalg.inv(scale_m)

    return CameraMotionResult(
        is_camera_motion=is_camera,
        homography=homography,
        good_matches=good_matches,
        kp1=kp1,
        kp2=kp2,
        vis_frame1=frame1,
        vis_frame2=frame2,
    )


def detect_camera_motion_orb(
    frame1: np.ndarray,
    frame2: np.ndarray,
    mask: np.ndarray | None = None,
    inlier_ratio_threshold: float = 0.75,
) -> bool:
    result = detect_camera_motion(
        frame1=frame1,
        frame2=frame2,
        mask=mask,
        feature_detector="ORB",
        inlier_ratio_threshold=inlier_ratio_threshold,
    )
    return result.is_camera_motion


def resize_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    if scale <= 0:
        return frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (max(1, width), max(1, height)), interpolation=cv2.INTER_AREA)


def visualize_homography(
    frame1: np.ndarray,
    frame2: np.ndarray,
    homography: np.ndarray,
    good_matches: list,
    kp1: list,
    kp2: list,
) -> np.ndarray:
    h, w = frame1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography)
    img2 = cv2.polylines(frame2.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    vis = cv2.drawMatches(frame1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis


def detect_motion(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    options: MotionOptions,
    previous_area: float,
    cutie_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    if options.method == "Optical Flow":
        motion_mask = _motion_mask_optical_flow(previous_frame, current_frame, options.use_cuda, cutie_mask)
    else:
        motion_mask = _motion_mask_absdiff(previous_frame, current_frame, cutie_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < options.min_area:
        return None, 0.0

    if previous_area and area < previous_area * options.area_ratio_threshold:
        return None, previous_area

    return contour, area


def _motion_mask_absdiff(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    cutie_mask: np.ndarray | None,
) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = _apply_mask(previous_gray, cutie_mask)
    current_gray = _apply_mask(current_gray, cutie_mask)
    diff = cv2.absdiff(previous_gray, current_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return motion_mask


def _motion_mask_optical_flow(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    use_cuda: bool,
    cutie_mask: np.ndarray | None,
) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = _apply_mask(previous_gray, cutie_mask)
    current_gray = _apply_mask(current_gray, cutie_mask)

    if use_cuda and _opencv_cuda_available():
        try:
            prev_gpu = cv2.cuda_GpuMat()
            curr_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(previous_gray)
            curr_gpu.upload(current_gray)
            cuda_flow = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=5,
                pyrScale=0.5,
                fastPyramids=False,
                winSize=21,
                numIters=10,
                polyN=7,
                polySigma=1.5,
                flags=0,
            )
            flow_gpu = cuda_flow.calc(prev_gpu, curr_gpu, None)
            flow = flow_gpu.download()
        except Exception as exc:
            log.warning("CUDA optical flow failed; falling back to CPU: %s", exc)
            flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    else:
        flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 21, 3, 7, 1.5, 0)

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = (magnitude > np.mean(magnitude)).astype(np.uint8) * 255
    return motion_mask


def draw_contour(frame: np.ndarray, contour: np.ndarray, cutie_mask: np.ndarray | None = None) -> np.ndarray:
    contour_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, 2)
    if cutie_mask is not None:
        contour_mask = np.where(cutie_mask > 0, contour_mask, 0).astype(np.uint8)
    out = frame.copy()
    out[contour_mask > 0] = [0, 255, 0]
    return out

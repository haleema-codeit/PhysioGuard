"""Video metadata and reliability measurements derived from decoded frames."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import ANALYSIS_SAMPLE_FPS, MAX_ANALYSIS_FRAMES, ROI_NAMES
from modules.face_detector import FaceDetector, FaceObservation


@dataclass
class VideoMetadata:
    path: str
    filename: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    frames_analyzed: int
    usable_frames: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityMetrics:
    face_visibility: float
    face_size: float
    lighting: float
    motion: float
    blur: float
    compression: float
    frame_stability: float
    usable_frames: float
    video_quality: float
    signal_reliability: float
    region_reliability: Dict[str, float]
    notes: List[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _score(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / max(1e-9, high - low) * 100.0, 0.0, 100.0))


def _determine_rotation(cap, first_frame: np.ndarray, detector: FaceDetector):
    meta_rot = 0
    try:
        meta_rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    except Exception:
        meta_rot = 0

    h, w = first_frame.shape[:2]
    if meta_rot == 90 and w > h:
        return cv2.ROTATE_90_CLOCKWISE
    if meta_rot == 270 and w > h:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if meta_rot == 180:
        return cv2.ROTATE_180

    if w > h:
        if detector.detect(first_frame) is None:
            rot90 = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
            if detector.detect(rot90) is not None:
                return cv2.ROTATE_90_CLOCKWISE
            rot270 = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if detector.detect(rot270) is not None:
                return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def _read_video(path: str, detector: FaceDetector, max_frames: int = MAX_ANALYSIS_FRAMES):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = ANALYSIS_SAMPLE_FPS
    stride = max(1, int(round(fps / ANALYSIS_SAMPLE_FPS)))
    frames, observations, timestamps = [], [], []
    i = 0
    rotation_code = None
    rotation_checked = False

    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if not rotation_checked:
            rotation_code = _determine_rotation(cap, frame, detector)
            rotation_checked = True

        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        if i % stride == 0:
            ts = i / fps
            frames.append(frame)
            timestamps.append(ts)
            observations.append(detector.observe(frame, i, ts))
        i += 1
    cap.release()

    width = frames[0].shape[1] if frames else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = frames[0].shape[0] if frames else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    metadata = VideoMetadata(path, path.split('\\')[-1].split('/')[-1], fps, frame_count,
                             width, height, frame_count / fps if fps else 0.0,
                             len(frames), sum(o.face is not None for o in observations))
    return metadata, frames, observations, timestamps


def analyze_video_quality(path: str, detector: Optional[FaceDetector] = None,
                          max_frames: int = MAX_ANALYSIS_FRAMES):
    detector = detector or FaceDetector()
    metadata, frames, observations, timestamps = _read_video(path, detector, max_frames)
    if not frames:
        raise ValueError("No decodable frames were found in the video")

    face_ratios = []
    face_sizes = []
    lighting_values = []
    blur_values = []
    motion_values = []
    compression_values = []
    centers = []
    for index, frame in enumerate(frames):
        obs = observations[index]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lighting_values.append(float(np.mean(gray)))
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_values.append(lap)
        if obs.face:
            face_ratios.append((obs.face.w * obs.face.h) / max(1, frame.shape[0] * frame.shape[1]))
            face_sizes.append((obs.face.w / max(1, frame.shape[1]) + obs.face.h / max(1, frame.shape[0])) / 2)
            centers.append((obs.face.x + obs.face.w / 2, obs.face.y + obs.face.h / 2, obs.face.w, obs.face.h))
        if index:
            prev = cv2.cvtColor(frames[index - 1], cv2.COLOR_BGR2GRAY)
            motion_values.append(float(np.mean(cv2.absdiff(gray, prev))))
        # High-frequency energy ratio is a useful, measurable compression proxy.
        dct_view = np.float32(gray[::4, ::4]) / 255.0
        dct_view = dct_view[:dct_view.shape[0] - dct_view.shape[0] % 2,
                            :dct_view.shape[1] - dct_view.shape[1] % 2]
        dct = cv2.dct(dct_view) if dct_view.size and min(dct_view.shape) >= 2 else np.zeros((2, 2), dtype=np.float32)
        compression_values.append(float(np.mean(np.abs(dct[8:, 8:])) if dct.shape[0] > 8 else 0.0))

    face_visibility = 100.0 * float(np.mean([o.face is not None for o in observations]))
    face_size = _score(float(np.mean(face_sizes)) if face_sizes else 0.0, 0.08, 0.35)
    mean_light = float(np.mean(lighting_values))
    lighting = max(0.0, 100.0 - abs(mean_light - 135.0) / 135.0 * 100.0)
    motion = max(0.0, 100.0 - _score(float(np.mean(motion_values)) if motion_values else 255.0, 2.0, 45.0))
    blur = _score(float(np.mean(blur_values)), 15.0, 350.0)
    compression = max(0.0, 100.0 - _score(float(np.mean(compression_values)), 0.002, 0.08))
    if len(centers) >= 2:
        c = np.asarray(centers, dtype=np.float64)
        movement = np.mean(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1) / np.maximum(c[:-1, 2], 1.0))
        frame_stability = max(0.0, 100.0 - _score(float(movement), 0.005, 0.18))
    else:
        frame_stability = 0.0
    usable = 100.0 * min(1.0, metadata.usable_frames / 120.0) * (face_visibility / 100.0)
    base_quality = float(np.mean([face_visibility, face_size, lighting, motion, blur, compression,
                                  frame_stability, usable]))
    # Region reliability is computed from actual coverage, local variance, and motion.
    region_reliability = {}
    for name in ROI_NAMES:
        coverage = np.mean([name in o.rois for o in observations]) * 100.0
        local_std = []
        for frame, obs in zip(frames, observations):
            if name in obs.rois:
                x, y, w, h = obs.rois[name]
                patch = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                if patch.size:
                    local_std.append(float(np.std(patch)))
        texture = _score(float(np.mean(local_std)) if local_std else 0.0, 4.0, 35.0)
        region_reliability[name] = float(np.clip(0.55 * coverage + 0.25 * blur + 0.20 * texture, 0, 100))
    signal_reliability = float(np.clip(0.55 * base_quality + 0.45 * np.mean(list(region_reliability.values())), 0, 100))
    notes = []
    if face_visibility < 60: notes.append("Face visibility is limited across the analyzed frames.")
    if motion < 45: notes.append("Motion is high enough to reduce physiological signal reliability.")
    if blur < 45: notes.append("Frames are soft or low-detail.")
    if metadata.usable_frames < 60: notes.append("The decoded sequence is short for robust rPPG analysis.")
    quality = QualityMetrics(face_visibility, face_size, lighting, motion, blur, compression,
                             frame_stability, usable, float(np.clip(base_quality, 0, 100)),
                             signal_reliability, region_reliability, notes)
    return metadata, frames, observations, timestamps, quality


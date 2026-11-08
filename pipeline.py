"""Orchestration layer used by the dashboard and evaluation runner."""

import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import PHYSIO_BAND, ROI_NAMES
from modules.consistency import calculate_consistency
from modules.explanation import explain
from modules.face_detector import FaceBox, FaceDetector, FaceObservation, draw_observation, roi_boxes
from modules.frame_analyzer import analyze_frames, build_temporal_segments, frame_vote_summary
from modules.fusion import fuse
from modules.rppg import extract_multi_region_rppg
from modules.video_quality import analyze_video_quality
from modules.visual_detector import VisualDetector


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_annotated_video(frames, observations, fps: float = 15.0):
    """Build a temporary MP4 clip where each frame is annotated with the tracked face overlay."""
    if not frames:
        return None
    tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    out_path = tmp_dir / f"annotated_clip_{abs(hash(str(frames[0].shape)))}.mp4"
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 15.0, (width, height))
    if not writer.isOpened():
        return None

    for frame, obs in zip(frames, observations):
        if obs is None:
            writer.write(frame)
            continue
        annotated = draw_observation(frame, obs)
        writer.write(annotated)
    writer.release()
    return str(out_path)


def _select_best_face_observation(observations):
    """Return the index of the most face-like detection, not simply the first valid one."""
    valid = [(idx, obs) for idx, obs in enumerate(observations) if obs is not None and obs.face is not None]
    if not valid:
        return 0

    def score(obs):
        face = obs.face
        if face is None:
            return float("-inf")
        frame_h, frame_w = obs.frame_shape
        area_ratio = (face.w * face.h) / max(1.0, frame_w * frame_h)
        aspect = face.w / max(1.0, face.h)
        center_y = (face.y + face.h / 2.0) / max(1.0, frame_h)
        center_x = (face.x + face.w / 2.0) / max(1.0, frame_w)
        height_ratio = face.h / max(1.0, frame_h)
        body_penalty = 0.0

        if frame_h > frame_w:
            body_penalty += max(0.0, center_y - 0.52) * 8.0
            body_penalty += max(0.0, height_ratio - 0.55) * 10.0
            body_penalty += max(0.0, (face.y + face.h) / max(1.0, frame_h) - 0.82) * 6.0
        score = 0.0
        score += 2.2 * min(1.0, area_ratio / 0.08)
        score += 2.0 * (1.0 - min(1.0, abs(aspect - 0.82) / 1.2))
        score += 1.8 * (1.0 - min(1.0, abs(center_y - 0.35) / 0.35))
        score += 1.0 * (1.0 - min(1.0, abs(center_x - 0.50) / 0.50))
        score -= body_penalty
        return score

    return max(valid, key=lambda item: score(item[1]))[0]


def _track_face_observations(observations):
    """Smooth the face box across adjacent frames so it follows a moving person."""
    smoothed = []
    prev_face = None
    for obs in observations:
        face = obs.face
        if face is not None and prev_face is not None:
            dx = abs(face.x - prev_face.x)
            dy = abs(face.y - prev_face.y)
            dw = abs(face.w - prev_face.w)
            dh = abs(face.h - prev_face.h)
            if (dx < prev_face.w * 0.75 and dy < prev_face.h * 0.75 and
                dw < prev_face.w * 0.70 and dh < prev_face.h * 0.70):
                merged_x = int(round(prev_face.x * 0.35 + face.x * 0.65))
                merged_y = int(round(prev_face.y * 0.35 + face.y * 0.65))
                merged_w = int(round(prev_face.w * 0.35 + face.w * 0.65))
                merged_h = int(round(prev_face.h * 0.35 + face.h * 0.65))
                face = FaceBox(merged_x, merged_y, max(1, merged_w), max(1, merged_h),
                              float((prev_face.score + face.score) / 2.0))
        if face is None and prev_face is not None:
            face = FaceBox(prev_face.x, prev_face.y, max(1, prev_face.w), max(1, prev_face.h), prev_face.score)

        if face is not None:
            rois = roi_boxes(face, obs.frame_shape[1], obs.frame_shape[0])
            obs = FaceObservation(obs.frame_index, obs.timestamp, obs.frame_shape, face, rois,
                                 obs.landmarks, obs.mesh_edges)
        smoothed.append(obs)
        prev_face = obs.face if obs.face is not None else prev_face

    return smoothed


def analyze_video(path: str, max_frames: int = 360, original_filename: str | None = None) -> dict:
    detector = FaceDetector()
    metadata, frames, observations, timestamps, quality = analyze_video_quality(path, detector, max_frames)
    observations = _track_face_observations(observations)
    # The reader samples the source video down to roughly 15 FPS.  The
    # container FPS is therefore not the sampling rate seen by the rPPG/FFT
    # code; extract_multi_region_rppg derives the effective rate from the
    # timestamps for each region.
    signals = extract_multi_region_rppg(frames, observations, timestamps,
                                        metadata.fps if metadata.fps > 0 else 15.0)
    consistency = calculate_consistency(signals, quality.region_reliability)
    first_face_index = _select_best_face_observation(observations)
    analyzed_frame = frames[first_face_index]
    observation = observations[first_face_index]
    visual = VisualDetector().analyze(analyzed_frame)
    fusion = fuse(visual.score, consistency.score, quality.signal_reliability,
                  quality.video_quality, consistency.reliable_regions, metadata.usable_frames,
                  quality.region_reliability)
    narrative = explain(fusion, consistency, quality, visual.model_status)
    annotated = draw_observation(analyzed_frame, observation)
    annotated_video = _encode_annotated_video(frames, observations, metadata.fps if metadata.fps > 0 else 15.0)
    file_size = Path(path).stat().st_size
    display_filename = original_filename or metadata.filename
    metadata_dict = metadata.as_dict()
    metadata_dict["filename"] = display_filename
    if len(timestamps) >= 2:
        deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
        deltas = deltas[np.isfinite(deltas) & (deltas > 1e-9)]
        metadata_dict["analysis_sample_fps"] = float(1.0 / np.median(deltas)) if len(deltas) else 15.0
    else:
        metadata_dict["analysis_sample_fps"] = 15.0

    # --- Per-frame analysis (reuses same stride as main pipeline) ---
    frame_records = analyze_frames(path, max_frames=max_frames)
    vote = frame_vote_summary(frame_records)
    segments = build_temporal_segments(frame_records)

    return {
        "metadata": metadata_dict, "quality": quality.as_dict(),
        "signals": {k: v.as_dict() for k, v in signals.items()},
        "consistency": consistency.as_dict(), "visual": visual.as_dict(),
        "fusion": fusion.as_dict(), "explanation": narrative,
        "preview_frame": analyzed_frame, "annotated_frame": annotated,
        "annotated_video": annotated_video,
        "heatmap": visual.heatmap, "edge_view": visual.edge_view,
        "frame_records": frame_records,
        "frame_vote": vote,
        "frame_segments": [s.as_dict() for s in segments],
        "provenance": {"filename": display_filename, "file_type": Path(path).suffix.lower(),
                        "file_size_bytes": file_size, "sha256": _sha256(path),
                        "analysis_timestamp": datetime.now().astimezone().isoformat()}
    }


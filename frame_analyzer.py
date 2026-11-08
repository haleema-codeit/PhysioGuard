"""Per-frame extraction and classification for PhysioGuard.

This module extracts every sampled frame (at the same stride used by the
main pipeline) and scores each one with the VisualDetector baseline.  It
does **not** duplicate the video-level inference: it reuses the existing
VisualDetector._baseline() for a fast, transparent, per-frame heuristic
score so that results can be shown alongside the overall fusion verdict.

Key design choices
------------------
* The same ANALYSIS_SAMPLE_FPS stride is used so extracted frames are
  identical to those seen by the rPPG / quality pipeline.
* Frames are stored as JPEG-compressed bytes in each FrameRecord to keep
  RAM bounded.  Full NumPy arrays are never accumulated across hundreds of
  frames simultaneously.
* Inference is fully batched in a single sequential pass – no duplicate
  reads or redundant processing.
* All per-frame results are returned as plain dicts for easy JSON export
  and Streamlit rendering.
"""

from __future__ import annotations

import io
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from config import ANALYSIS_SAMPLE_FPS, MAX_ANALYSIS_FRAMES
from modules.visual_detector import VisualDetector

# Thresholds for per-frame labelling. The fallback visual detector is a
# transparent pixel heuristic, not a trained classifier, so it should be
# conservative and only label frames as FAKE when the signal is clearly
# abnormal. Ordinary real faces should stay in the REAL/UNCERTAIN band.
_FAKE_THRESHOLD = 75.0   # score >= this → FAKE
_REAL_THRESHOLD = 35.0   # score <= this → REAL
# Scores between the two thresholds → UNCERTAIN

_JPEG_QUALITY = 70        # JPEG compression quality for stored thumbnails
_DISPLAY_MAX_DIM = 480    # longest-edge pixels for stored display image


@dataclass
class FrameRecord:
    """A single extracted frame with its classification result."""

    frame_index: int          # sequential index among extracted frames (0-based)
    source_frame_index: int   # original video frame number
    timestamp: float          # video timestamp in seconds
    label: str                # "REAL" | "FAKE" | "UNCERTAIN" | "FAILED"
    fake_score: Optional[float]   # 0-100; higher = more likely fake
    real_score: Optional[float]   # 0-100; higher = more likely real
    confidence: Optional[float]   # 0-100 derived from distance to threshold
    model_status: str         # text description of which model produced the score
    details: dict             # model-specific diagnostic values
    status: str               # "ok" | "decode_error" | "inference_error"
    error_message: Optional[str]
    # Compressed frame image stored as bytes (JPEG).  None on decode failure.
    image_bytes: Optional[bytes] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "source_frame_index": self.source_frame_index,
            "timestamp": self.timestamp,
            "label": self.label,
            "fake_score": self.fake_score,
            "real_score": self.real_score,
            "confidence": self.confidence,
            "model_status": self.model_status,
            "details": self.details,
            "status": self.status,
            "error_message": self.error_message,
        }


def _compress_frame(frame: np.ndarray) -> Optional[bytes]:
    """Resize and JPEG-compress a BGR frame. Returns None on error."""
    try:
        h, w = frame.shape[:2]
        if max(h, w) > _DISPLAY_MAX_DIM:
            scale = _DISPLAY_MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return bytes(buf) if ok else None
    except Exception:
        return None


def _label_from_score(score: float) -> str:
    if score >= _FAKE_THRESHOLD:
        return "FAKE"
    if score <= _REAL_THRESHOLD:
        return "REAL"
    return "UNCERTAIN"


def _confidence_from_score(score: float) -> float:
    """Map a 0-100 score to a 50-100 confidence via distance to 50."""
    return float(np.clip(50.0 + abs(score - 50.0) * 1.2, 0.0, 100.0))


def analyze_frames(
    video_path: str,
    max_frames: int = MAX_ANALYSIS_FRAMES,
    sample_fps: float = ANALYSIS_SAMPLE_FPS,
) -> List[FrameRecord]:
    """Extract and classify every sampled frame in the video.

    Parameters
    ----------
    video_path:
        Absolute path to the video file.
    max_frames:
        Maximum number of frames to extract (same cap as the main pipeline).
    sample_fps:
        Target sampling rate in frames-per-second.

    Returns
    -------
    A list of :class:`FrameRecord` objects, one per extracted frame, in
    temporal order.
    """
    detector = VisualDetector()
    records: List[FrameRecord] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return records

    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

    meta_rot = 0
    try:
        meta_rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    except Exception:
        meta_rot = 0

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = sample_fps

    stride = max(1, int(round(source_fps / sample_fps)))

    source_frame_no = 0
    extracted_idx = 0
    rotation_code = None
    rotation_checked = False

    while extracted_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if not rotation_checked:
            h, w = frame.shape[:2]
            if meta_rot == 90 and w > h:
                rotation_code = cv2.ROTATE_90_CLOCKWISE
            elif meta_rot == 270 and w > h:
                rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif meta_rot == 180:
                rotation_code = cv2.ROTATE_180
            rotation_checked = True

        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        if source_frame_no % stride == 0:
            timestamp = source_frame_no / source_fps
            image_bytes: Optional[bytes] = None
            label = "FAILED"
            fake_score: Optional[float] = None
            real_score: Optional[float] = None
            confidence: Optional[float] = None
            model_status = detector.model_status
            details: dict = {}
            status = "ok"
            error_msg: Optional[str] = None

            try:
                image_bytes = _compress_frame(frame)
            except Exception as exc:
                status = "decode_error"
                error_msg = str(exc)

            if status == "ok":
                try:
                    visual_result = detector.analyze(frame)
                    fake_score = float(visual_result.score)
                    real_score = float(np.clip(100.0 - fake_score, 0.0, 100.0))
                    label = _label_from_score(fake_score)
                    confidence = _confidence_from_score(fake_score)
                    model_status = visual_result.model_status
                    details = visual_result.details or {}
                except Exception as exc:
                    status = "inference_error"
                    error_msg = str(exc)
                    label = "FAILED"

            records.append(FrameRecord(
                frame_index=extracted_idx,
                source_frame_index=source_frame_no,
                timestamp=timestamp,
                label=label,
                fake_score=fake_score,
                real_score=real_score,
                confidence=confidence,
                model_status=model_status,
                details=details,
                status=status,
                error_message=error_msg,
                image_bytes=image_bytes,
            ))
            extracted_idx += 1

        source_frame_no += 1

    cap.release()
    return records


# ---------------------------------------------------------------------------
# Temporal segmentation
# ---------------------------------------------------------------------------

@dataclass
class TemporalSegment:
    """A consecutive run of frames sharing the same broad classification."""

    label: str                # "REAL" | "FAKE" | "UNCERTAIN"
    start_frame_index: int    # among extracted frames
    end_frame_index: int
    start_timestamp: float
    end_timestamp: float
    frame_count: int
    fake_scores: List[float]  # all fake scores within the segment
    source_frame_start: int
    source_frame_end: int

    @property
    def mean_fake_score(self) -> Optional[float]:
        return float(np.mean(self.fake_scores)) if self.fake_scores else None

    @property
    def max_fake_score(self) -> Optional[float]:
        return float(np.max(self.fake_scores)) if self.fake_scores else None

    @property
    def min_fake_score(self) -> Optional[float]:
        return float(np.min(self.fake_scores)) if self.fake_scores else None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "frame_count": self.frame_count,
            "source_frame_start": self.source_frame_start,
            "source_frame_end": self.source_frame_end,
            "mean_fake_score": self.mean_fake_score,
            "max_fake_score": self.max_fake_score,
            "min_fake_score": self.min_fake_score,
        }


def _smooth_labels(labels: List[str], window: int = 3) -> List[str]:
    """Majority-vote smoothing to suppress isolated noisy predictions."""
    if len(labels) <= window:
        return labels
    smoothed = list(labels)
    half = window // 2
    for i in range(half, len(labels) - half):
        window_labels = labels[i - half: i + half + 1]
        for candidate in ("FAKE", "REAL", "UNCERTAIN"):
            if window_labels.count(candidate) > window // 2:
                smoothed[i] = candidate
                break
    return smoothed


def build_temporal_segments(
    records: List[FrameRecord],
    smoothing_window: int = 5,
    min_segment_frames: int = 2,
) -> List[TemporalSegment]:
    """Group consecutive frames with the same predicted label into segments.

    FAILED frames are excluded from segments; only REAL / FAKE / UNCERTAIN
    frames participate.  Short isolated runs shorter than *min_segment_frames*
    are merged with their neighbours.
    """
    classified = [r for r in records if r.label in ("REAL", "FAKE", "UNCERTAIN")]
    if not classified:
        return []

    labels = [r.label for r in classified]
    labels = _smooth_labels(labels, window=smoothing_window)

    # Build raw runs.
    segments: List[TemporalSegment] = []
    run_start = 0
    for i in range(1, len(classified)):
        if labels[i] != labels[i - 1]:
            seg_records = classified[run_start:i]
            fake_scores_in_seg = [
                r.fake_score for r in seg_records if r.fake_score is not None
            ]
            segments.append(TemporalSegment(
                label=labels[run_start],
                start_frame_index=seg_records[0].frame_index,
                end_frame_index=seg_records[-1].frame_index,
                start_timestamp=seg_records[0].timestamp,
                end_timestamp=seg_records[-1].timestamp,
                frame_count=len(seg_records),
                fake_scores=fake_scores_in_seg,
                source_frame_start=seg_records[0].source_frame_index,
                source_frame_end=seg_records[-1].source_frame_index,
            ))
            run_start = i

    # Append the final run.
    seg_records = classified[run_start:]
    fake_scores_in_seg = [
        r.fake_score for r in seg_records if r.fake_score is not None
    ]
    segments.append(TemporalSegment(
        label=labels[run_start],
        start_frame_index=seg_records[0].frame_index,
        end_frame_index=seg_records[-1].frame_index,
        start_timestamp=seg_records[0].timestamp,
        end_timestamp=seg_records[-1].timestamp,
        frame_count=len(seg_records),
        fake_scores=fake_scores_in_seg,
        source_frame_start=seg_records[0].source_frame_index,
        source_frame_end=seg_records[-1].source_frame_index,
    ))

    # Merge short segments into their dominant neighbour.
    if len(segments) > 1:
        merged = True
        while merged and len(segments) > 1:
            merged = False
            new_segs: List[TemporalSegment] = []
            skip = False
            for idx in range(len(segments)):
                if skip:
                    skip = False
                    continue
                seg = segments[idx]
                if seg.frame_count < min_segment_frames and idx + 1 < len(segments):
                    # Absorb into next segment.
                    nxt = segments[idx + 1]
                    combined_scores = seg.fake_scores + nxt.fake_scores
                    new_segs.append(TemporalSegment(
                        label=nxt.label,
                        start_frame_index=seg.start_frame_index,
                        end_frame_index=nxt.end_frame_index,
                        start_timestamp=seg.start_timestamp,
                        end_timestamp=nxt.end_timestamp,
                        frame_count=seg.frame_count + nxt.frame_count,
                        fake_scores=combined_scores,
                        source_frame_start=seg.source_frame_start,
                        source_frame_end=nxt.source_frame_end,
                    ))
                    skip = True
                    merged = True
                else:
                    new_segs.append(seg)
            segments = new_segs

    return segments


# ---------------------------------------------------------------------------
# Frame-vote summary for combine_fusion_and_frame_vote
# ---------------------------------------------------------------------------

def frame_vote_summary(records: List[FrameRecord]) -> dict:
    """Return a summary dict compatible with combine_fusion_and_frame_vote."""
    fake_count = sum(1 for r in records if r.label == "FAKE")
    real_count = sum(1 for r in records if r.label == "REAL")
    uncertain_count = sum(1 for r in records if r.label == "UNCERTAIN")
    failed_count = sum(1 for r in records if r.label == "FAILED")
    scored = [r.fake_score for r in records if r.fake_score is not None]
    return {
        "total_frames": len(records),
        "fake_count": fake_count,
        "real_count": real_count,
        "uncertain_count": uncertain_count,
        "failed_count": failed_count,
        "mean_fake_score": float(np.mean(scored)) if scored else None,
        "max_fake_score": float(np.max(scored)) if scored else None,
        "min_fake_score": float(np.min(scored)) if scored else None,
    }

"""Face detection and relative physiological ROI extraction.

MediaPipe is used when available; OpenCV's shipped Haar cascade is the
portable fallback. Both paths return the same normalized face/ROI contract.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from config import DEFAULT_FACE_CASCADE, ROI_COLORS, ROI_NAMES


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaceObservation:
    frame_index: int
    timestamp: float
    frame_shape: Tuple[int, int]
    face: Optional[FaceBox]
    rois: Dict[str, Tuple[int, int, int, int]]
    landmarks: List[Tuple[int, int]]
    mesh_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]


def _clip_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(x1 + 1, min(width, int(round(x + w))))
    y2 = max(y1 + 1, min(height, int(round(y + h))))
    return x1, y1, x2 - x1, y2 - y1


def roi_boxes(face: FaceBox, width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
    """Return conservative inner-face ROIs as x,y,w,h pixel boxes."""
    x, y, w, h = face.x, face.y, face.w, face.h
    relative = {
        "Forehead": (0.24, 0.12, 0.52, 0.20),
        "Left Cheek": (0.10, 0.48, 0.29, 0.25),
        "Right Cheek": (0.61, 0.48, 0.29, 0.25),
    }
    return {name: _clip_box((x + rx * w, y + ry * h, rw * w, rh * h), width, height)
            for name, (rx, ry, rw, rh) in relative.items()}


class FaceDetector:
    def __init__(self, cascade_path: Optional[str] = None):
        cascade_path = cascade_path or (cv2.data.haarcascades + DEFAULT_FACE_CASCADE)
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Unable to load face detector cascade: {cascade_path}")
        self.mesh = None
        self.mesh_connections = []
        self.face_detection_full = None
        self.face_detection_short = None
        try:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self.mesh_connections = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)
            # Full-range (up to 5m, great for portraits & medium shots)
            self.face_detection_full = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.35
            )
            # Short-range (up to 2m, optimized for selfies / close-up mobile video)
            self.face_detection_short = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.35
            )
        except Exception:
            self.mesh = None
            self.face_detection_full = None
            self.face_detection_short = None

    @staticmethod
    def _choose_best_face(candidates, width: int, height: int):
        """Pick the most face-like candidate and reject torso/body false positives."""
        if not candidates:
            return None

        best = None
        best_score = float("-inf")
        frame_area = max(1, width * height)
        center_x = width / 2.0
        is_portrait = height > width

        # In portrait videos, the human head naturally sits in the upper 20%-45%
        # of the frame; vertical center (50%) corresponds to chest/torso.
        expected_cy = 0.35 * height if is_portrait else 0.45 * height

        for x, y, w, h in candidates:
            if w <= 0 or h <= 0:
                continue
            area = float(w * h)
            aspect = float(w) / max(1.0, float(h))

            # Filter candidates that are tiny specks or excessively massive
            if area < 0.0020 * frame_area or area > 0.75 * frame_area:
                continue
            if h > 0.85 * height or w > 0.95 * width:
                continue
            if is_portrait and h > 0.62 * height:
                continue
            # Avoid candidates hugging the bottom edge (clothing / torso)
            if y + h > 0.97 * height and area > 0.15 * frame_area:
                continue
            if is_portrait and (y + h) > 0.82 * height and area > 0.08 * frame_area:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            horiz_penalty = abs(cx - center_x) / max(1.0, width)
            vert_dist = abs(cy - expected_cy) / max(1.0, height)

            # In portrait videos, strongly penalize detections located down in the lower torso/abdomen
            lower_body_penalty = 0.0
            if is_portrait:
                lower_body_penalty = max(0.0, (cy / height) - 0.45) * 7.0
                if (cy / height) > 0.58:
                    lower_body_penalty += ((cy / height) - 0.58) * 12.0

            aspect_score = 1.0 - abs(aspect - 0.82) / 1.35
            size_score = min(1.0, area / max(1.0, 0.06 * frame_area))
            vertical_score = 1.0 - min(1.0, vert_dist / 0.50)

            score = (
                1.5 * size_score
                + 1.3 * aspect_score
                + 1.2 * vertical_score
                - 1.4 * horiz_penalty
                - lower_body_penalty
            )
            if score > best_score:
                best = (int(x), int(y), int(w), int(h))
                best_score = score
        return best

    def _detect_mediapipe(self, rgb: np.ndarray, width: int, height: int) -> Optional[FaceBox]:
        """Detect face using MediaPipe deep-learning face detectors."""
        for detector in (self.face_detection_full, self.face_detection_short):
            if detector is None:
                continue
            try:
                res = detector.process(rgb)
                if res and res.detections:
                    candidates = []
                    for d in res.detections:
                        bb = d.location_data.relative_bounding_box
                        score = float(d.score[0]) if d.score else 1.0
                        bx = int(round(bb.xmin * width))
                        by = int(round(bb.ymin * height))
                        bw = int(round(bb.width * width))
                        bh = int(round(bb.height * height))

                        x1 = max(0, min(width - 1, bx))
                        y1 = max(0, min(height - 1, by))
                        x2 = max(x1 + 1, min(width, bx + bw))
                        y2 = max(y1 + 1, min(height, by + bh))
                        candidates.append((x1, y1, x2 - x1, y2 - y1, score))

                    if candidates:
                        best = max(candidates, key=lambda c: c[4])
                        return FaceBox(best[0], best[1], best[2], best[3], best[4])
            except Exception:
                continue
        return None

    def _detect_from_mesh(self, rgb: np.ndarray, width: int, height: int) -> Tuple[Optional[FaceBox], List[Tuple[int, int]]]:
        """Detect face directly from MediaPipe FaceMesh landmarks."""
        if self.mesh is None:
            return None, []
        try:
            res = self.mesh.process(rgb)
            if res and res.multi_face_landmarks:
                pts = [(int(round(pt.x * width)), int(round(pt.y * height)))
                       for pt in res.multi_face_landmarks[0].landmark]
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    min_x, max_x = max(0, min(xs)), min(width, max(xs))
                    min_y, max_y = max(0, min(ys)), min(height, max(ys))
                    bw = max_x - min_x
                    bh = max_y - min_y
                    pad_x = int(bw * 0.05)
                    pad_y = int(bh * 0.08)
                    fx = max(0, min_x - pad_x)
                    fy = max(0, min_y - pad_y)
                    fw = min(width - fx, bw + 2 * pad_x)
                    fh = min(height - fy, bh + 2 * pad_y)
                    return FaceBox(fx, fy, max(1, fw), max(1, fh), 1.0), pts
        except Exception:
            pass
        return None, []

    def _detect_rotated(self, frame: np.ndarray, width: int, height: int) -> Optional[FaceBox]:
        """Try 90° CW, 270° CW, and 180° rotations if upright detection failed (e.g. rotated mobile video)."""
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
            (cv2.ROTATE_180, 180),
        ]
        for rot_code, angle in rotations:
            try:
                rot_frame = cv2.rotate(frame, rot_code)
                rw, rh = rot_frame.shape[1], rot_frame.shape[0]
                rgb = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2RGB)
                face = self._detect_mediapipe(rgb, rw, rh)
                if not face:
                    face, _ = self._detect_from_mesh(rgb, rw, rh)
                if face:
                    if angle == 90:
                        orig_x = face.y
                        orig_y = height - (face.x + face.w)
                        orig_w = face.h
                        orig_h = face.w
                    elif angle == 270:
                        orig_x = width - (face.y + face.h)
                        orig_y = face.x
                        orig_w = face.h
                        orig_h = face.w
                    else:
                        orig_x = width - (face.x + face.w)
                        orig_y = height - (face.y + face.h)
                        orig_w = face.w
                        orig_h = face.h
                    return FaceBox(max(0, orig_x), max(0, orig_y), max(1, orig_w), max(1, orig_h), face.score)
            except Exception:
                continue
        return None

    def _detect_haar(self, frame: np.ndarray, width: int, height: int) -> Optional[FaceBox]:
        """Aspect-ratio-invariant Haar cascade fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_dim = max(24, min(width, height) // 16)
        max_w = int(width * 0.98)
        max_h = int(height * 0.98)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(min_dim, min_dim),
            maxSize=(max_w, max_h)
        )
        if len(faces) == 0:
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(min_dim, min_dim),
                maxSize=(max_w, max_h)
            )
        if len(faces) == 0:
            return None

        best = self._choose_best_face(faces.tolist(), width, height)
        if best is None:
            x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
            return FaceBox(int(x), int(y), int(w), int(h), 1.0)
        x, y, w, h = best
        return FaceBox(int(x), int(y), int(w), int(h), 1.0)

    def detect(self, frame: np.ndarray) -> Optional[FaceBox]:
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Primary: MediaPipe Face Detection
        face = self._detect_mediapipe(rgb, width, height)
        if face is not None:
            return face

        # 2. Secondary: MediaPipe FaceMesh
        face, _ = self._detect_from_mesh(rgb, width, height)
        if face is not None:
            return face

        # 3. Check for rotated video orientation if not found upright
        face = self._detect_rotated(frame, width, height)
        if face is not None:
            return face

        # 4. Fallback: Haar Cascade
        return self._detect_haar(frame, width, height)

    def observe(self, frame: np.ndarray, frame_index: int, timestamp: float) -> FaceObservation:
        if frame is None or frame.size == 0:
            return FaceObservation(frame_index, timestamp, (0, 0), None, {}, [], [])
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = None
        landmarks = []
        mesh_edges = []

        # 1. Attempt FaceMesh first to obtain rich 3D landmarks + accurate face box
        if self.mesh is not None:
            face, landmarks = self._detect_from_mesh(rgb, width, height)

        # 2. If FaceMesh did not find a face, run full detection pipeline
        if face is None:
            face = self.detect(frame)

        # 3. If face found but landmarks empty (e.g. from FaceDetection or Haar fallback),
        # try FaceMesh once more or populate deterministic mesh guides
        if face and not landmarks:
            if self.mesh is not None:
                try:
                    res = self.mesh.process(rgb)
                    if res and res.multi_face_landmarks:
                        landmarks = [(int(round(pt.x * width)), int(round(pt.y * height)))
                                     for pt in res.multi_face_landmarks[0].landmark]
                except Exception:
                    pass
            if not landmarks:
                for gx in np.linspace(0.1, 0.9, 9):
                    for gy in np.linspace(0.12, 0.9, 9):
                        landmarks.append((int(face.x + gx * face.w), int(face.y + gy * face.h)))

        if self.mesh_connections and landmarks and len(landmarks) > max(max(edge) for edge in self.mesh_connections):
            mesh_edges = [(landmarks[a], landmarks[b]) for a, b in self.mesh_connections]

        rois = roi_boxes(face, width, height) if face else {}
        return FaceObservation(frame_index, timestamp, (height, width), face, rois, landmarks, mesh_edges)


def draw_observation(frame: np.ndarray, observation: FaceObservation) -> np.ndarray:
    """Draw the actual detected face, geometric mesh guides, and three ROIs."""
    canvas = frame.copy()
    if observation.face:
        f = observation.face
        cv2.rectangle(canvas, (f.x, f.y), (f.x + f.w, f.y + f.h), (226, 237, 247), 2)
        for px, py in observation.landmarks:
            cv2.circle(canvas, (px, py), 1, (100, 210, 230), -1)
        # Connect grid guides only inside the face box.
        if len(observation.landmarks) == 81:
            for row in range(9):
                pts = observation.landmarks[row * 9:(row + 1) * 9]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(canvas, a, b, (90, 170, 195), 1)
            for col in range(9):
                pts = observation.landmarks[col::9]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(canvas, a, b, (90, 170, 195), 1)
        else:
            for a, b in observation.mesh_edges:
                cv2.line(canvas, a, b, (90, 170, 195), 1)
    for name in ROI_NAMES:
        if name not in observation.rois:
            continue
        x, y, w, h = observation.rois[name]
        color = ROI_COLORS[name]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, name.upper(), (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, color, 1, cv2.LINE_AA)
    return canvas


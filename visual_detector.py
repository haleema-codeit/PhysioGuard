"""Independent visual evidence module.

An optional Hugging Face image-classification model can be enabled with
PHYSIOGUARD_VISUAL_MODEL. When it is unavailable, the module still computes a
transparent, non-trained forensic artifact score from decoded pixels. That
fallback is explicitly labeled in the returned model status and UI.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np


@dataclass
class VisualResult:
    score: float
    model_status: str
    model_name: str
    heatmap: Optional[np.ndarray]
    edge_view: Optional[np.ndarray]
    details: dict

    def as_dict(self):
        data = asdict(self)
        data["heatmap"] = None
        data["edge_view"] = None
        return data


class VisualDetector:
    def __init__(self):
        self.model_name = os.getenv("PHYSIOGUARD_VISUAL_MODEL", "")
        self.classifier = None
        self.model_status = "PIXEL-FORENSIC BASELINE"
        if self.model_name:
            try:
                from transformers import pipeline
                self.classifier = pipeline("image-classification", model=self.model_name)
                self.model_status = "PRETRAINED MODEL READY"
            except Exception as exc:
                self.model_status = f"PRETRAINED MODEL UNAVAILABLE: {type(exc).__name__}"

    def _baseline(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # The fallback detector should not mark ordinary real faces as fake just
        # because the frame is smooth or slightly low-contrast. Instead, reward
        # natural texture and edge content, and only rise sharply when the frame
        # is effectively flat or visually inconsistent.
        block = gray[:gray.shape[0] - gray.shape[0] % 8, :gray.shape[1] - gray.shape[1] % 8]
        blocks = block.reshape(block.shape[0] // 8, 8, block.shape[1] // 8, 8).transpose(0, 2, 1, 3)
        block_std = np.std(blocks, axis=(2, 3))
        spatial_variation = float(np.std(block_std))
        edge_density = float(np.mean(edges > 0))

        # Natural frames keep some texture and edge structure; near-constant or
        # over-smoothed frames are more suspicious, but the heuristic should not
        # overreact to ordinary real faces. Keep the calibration conservative.
        texture_score = float(np.clip(spatial_variation / 26.0, 0.0, 1.0))
        edge_score = float(np.clip(edge_density * 10.0, 0.0, 1.0))
        blur_penalty = float(np.clip(max(0.0, 400.0 - lap) / 400.0, 0.0, 1.0))

        # This is intentionally a weak artifact signal, not a true classifier.
        score = float(np.clip(100.0 * (
            0.25 * (1.0 - texture_score) +
            0.25 * (1.0 - edge_score) +
            0.15 * blur_penalty
        ), 0.0, 100.0))

        heat = cv2.normalize(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        return score, heat, edges, {"laplacian_variance": lap, "block_variation": spatial_variation,
                                     "edge_density": edge_density, "texture_score": texture_score,
                                     "edge_score": edge_score, "blur_penalty": blur_penalty}

    def analyze(self, frame: np.ndarray) -> VisualResult:
        if self.classifier is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictions = self.classifier(rgb, top_k=5)
                fake = 0.0
                for item in predictions:
                    label = str(item.get("label", "")).lower()
                    if any(word in label for word in ("fake", "deepfake", "synthetic", "manipulated")):
                        fake += float(item.get("score", 0.0))
                heat = cv2.cvtColor(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 160), cv2.COLOR_GRAY2BGR)
                return VisualResult(float(np.clip(fake * 100, 0, 100)), self.model_status,
                                    self.model_name, heat, cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 160),
                                    {"predictions": predictions})
            except Exception as exc:
                self.model_status = f"MODEL ERROR; BASELINE USED: {type(exc).__name__}"
        score, heat, edges, details = self._baseline(frame)
        return VisualResult(score, self.model_status, "pixel-forensic-baseline", heat, edges, details)


"""Quality-aware evidence fusion and conservative three-level verdict."""

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class FusionResult:
    verdict: str
    synthetic_probability: Optional[float]
    final_evidence: float
    visual_manipulation: float
    physiological_consistency: float
    physiological_inconsistency: float
    quality_normalized_signal_gap: float
    signal_reliability: float
    video_quality: float
    physiological_weight: float
    visual_weight: float
    confidence: Optional[float]
    reasons: list
    decision_basis: str

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def as_dict(self):
        return asdict(self)


def combine_fusion_and_frame_vote(fusion: FusionResult | dict | None, frame_vote: dict | None) -> str:
    """Combine the current fusion verdict with the per-frame majority vote.

    The final decision is not based on a single override: all available evidence
    is merged so a mixed real/fake clip can still be judged as fake when the
    majority of frames and the underlying evidence support that outcome.
    """
    if fusion is None:
        return "INCONCLUSIVE"
    if not isinstance(fusion, dict):
        fusion_dict = fusion.as_dict() if hasattr(fusion, "as_dict") else {"verdict": getattr(fusion, "verdict", "INCONCLUSIVE")}
    else:
        fusion_dict = fusion

    if frame_vote is None:
        return str(fusion_dict.get("verdict", "INCONCLUSIVE"))

    total = max(int(frame_vote.get("total_frames", 0)), 1)
    fake_count = int(frame_vote.get("fake_count", 0))
    real_count = int(frame_vote.get("real_count", 0))
    fake_ratio = fake_count / total
    real_ratio = real_count / total
    frame_bias = (fake_ratio - real_ratio) * 100.0

    base_score = float(fusion_dict.get("final_evidence", 50.0))
    signal_reliability = float(fusion_dict.get("signal_reliability", 0.0))
    video_quality = float(fusion_dict.get("video_quality", 0.0))

    if signal_reliability < 30 or video_quality < 35:
        base_score -= 10.0

    combined_score = base_score + frame_bias * 0.8

    if combined_score >= 60.0:
        return "POTENTIAL DEEPFAKE"
    if combined_score <= 40.0:
        return "AUTHENTIC"
    return "INCONCLUSIVE"


def fuse(visual_score: float, consistency_score: float, signal_reliability: float,
         video_quality: float, reliable_regions: int, usable_frames: int,
         region_reliability: dict | None = None) -> FusionResult:
    raw_inconsistency = float(np.clip(100.0 - consistency_score, 0, 100))
    # A physiological signal that is unexpectedly weak relative to otherwise
    # usable video is a weak anomaly clue. It is never used by itself: motion,
    # lighting, makeup, occlusion, and skin appearance can all cause this gap.
    quality_normalized_gap = float(np.clip((video_quality - signal_reliability) * 2.0, 0, 100))
    phys_inconsistency = float(np.clip(0.80 * raw_inconsistency + 0.20 * quality_normalized_gap, 0, 100))
    quality_factor = float(np.clip(signal_reliability / 100.0, 0, 1))
    visual_factor = float(np.clip(0.55 + 0.45 * video_quality / 100.0, 0, 1))
    raw_visual_weight = 0.55 * visual_factor
    raw_phys_weight = 0.45 * quality_factor
    denom = max(raw_visual_weight + raw_phys_weight, 1e-9)
    visual_weight = raw_visual_weight / denom
    physiological_weight = raw_phys_weight / denom
    final = float(np.clip(visual_weight * visual_score + physiological_weight * phys_inconsistency, 0, 100))
    evidence_sufficient = reliable_regions >= 2 and signal_reliability >= 40 and usable_frames >= 30
    reasons = []
    if reliable_regions < 2: reasons.append("fewer than two reliable facial regions")
    if signal_reliability < 40: reasons.append("low signal reliability")
    if usable_frames < 30: reasons.append("too few usable frames")
    conflict = (visual_score >= 70 and consistency_score >= 70) or (visual_score < 35 and consistency_score < 35)
    suspicious_signal_degradation = (
        video_quality >= 55 and signal_reliability >= 40 and quality_normalized_gap >= 5
        and visual_score >= 50 and consistency_score <= 65
    )
    region_rule_active = region_reliability is not None and len(region_reliability) >= 3
    low_regions = [name for name, value in (region_reliability or {}).items() if float(value) < 70.0]
    if region_rule_active:
        # Explicit demo rule requested for this prototype: region reliability
        # alone determines the top-level verdict. The underlying evidence
        # scores remain calculated and displayed for inspection.
        verdict = "POTENTIAL DEEPFAKE" if low_regions else "AUTHENTIC"
        probability = final
        confidence = float(np.clip(50 + abs(final - 50) * 1.3, 0, 99))
        reasons = (["region reliability below 70%: " + ", ".join(low_regions)]
                   if low_regions else ["all three facial regions meet the 70% reliability rule"])
        decision_basis = "EXPLICIT REGION RELIABILITY RULE"
    elif not evidence_sufficient or conflict:
        verdict = "INCONCLUSIVE"
        probability = None
        confidence = None
        if conflict: reasons.append("visual and physiological evidence are contradictory")
        decision_basis = "QUALITY-AWARE EVIDENCE FUSION"
    # The calibrated evidence scale produced by the current visual baseline
    # clusters ordinary clips around the middle of 0-100. Use the actual
    # evidence components as tie-breakers so every good-quality clip is not
    # trapped in the uncertainty band, while retaining a real inconclusive
    # path for weak/contradictory evidence.
    elif final >= 60 or (visual_score >= 62 and consistency_score <= 55) or suspicious_signal_degradation:
        verdict = "POTENTIAL DEEPFAKE"
        probability = final
        confidence = float(np.clip(50 + abs(final - 50) * 1.3, 0, 99))
        decision_basis = "QUALITY-AWARE EVIDENCE FUSION"
    elif final <= 55 or (consistency_score >= 65 and visual_score < 65):
        verdict = "AUTHENTIC"
        probability = final
        confidence = float(np.clip(50 + abs(final - 50) * 1.3, 0, 99))
        decision_basis = "QUALITY-AWARE EVIDENCE FUSION"
    else:
        verdict = "INCONCLUSIVE"
        probability = None
        confidence = None
        reasons.append("evidence falls in the system's uncertainty band")
        decision_basis = "QUALITY-AWARE EVIDENCE FUSION"
    return FusionResult(verdict, probability, final, visual_score, consistency_score, phys_inconsistency,
                        quality_normalized_gap,
                        signal_reliability, video_quality, physiological_weight, visual_weight,
                        confidence, reasons, decision_basis)


"""Evidence-grounded narrative generation."""

from modules.fusion import FusionResult
from modules.consistency import ConsistencyResult
from modules.video_quality import QualityMetrics


def explain(fusion: FusionResult, consistency: ConsistencyResult, quality: QualityMetrics,
            visual_status: str) -> dict:
    bullets = []
    if fusion.decision_basis == "EXPLICIT REGION RELIABILITY RULE":
        below = [f"{name}: {value:.0f}%" for name, value in quality.region_reliability.items() if value < 70]
        if fusion.verdict == "POTENTIAL DEEPFAKE":
            bullets.append("The explicit region-reliability rule was triggered.")
            bullets.append("Below threshold: " + ", ".join(below) + ".")
            conclusion = "At least one facial physiological region is below the configured 70% reliability threshold, so this prototype labels the clip POTENTIAL DEEPFAKE."
        else:
            bullets.append("Forehead, left cheek, and right cheek all met the configured 70% reliability threshold.")
            conclusion = "All three facial regions meet the configured region-reliability rule, so this prototype labels the clip AUTHENTIC."
    elif fusion.verdict == "POTENTIAL DEEPFAKE":
        if fusion.visual_manipulation >= 60:
            bullets.append(f"Visual manipulation evidence was elevated ({fusion.visual_manipulation:.0f}/100).")
        if consistency.score < 55 and consistency.reliable_regions >= 2:
            bullets.append(f"Cross-region physiological consistency was low ({consistency.score:.0f}/100).")
        if fusion.signal_reliability >= 55:
            bullets.append("Signal reliability was sufficient for physiological evidence to contribute.")
        if fusion.quality_normalized_signal_gap >= 5:
            bullets.append(f"Signal reliability trailed overall video quality by a measured gap ({fusion.quality_normalized_signal_gap:.0f}/100 anomaly contribution).")
        conclusion = "Multiple evidence sources indicate possible manipulation; this is an AI-generated screening result, not a legal finding."
    elif fusion.verdict == "AUTHENTIC":
        bullets.append(f"Visual manipulation evidence was limited ({fusion.visual_manipulation:.0f}/100).")
        bullets.append(f"Reliable regions showed compatible physiological behavior ({consistency.score:.0f}/100 consistency).")
        conclusion = "The available evidence is sufficiently compatible with an authentic video, subject to the limits of the analyzed clip."
    else:
        if quality.motion < 50: bullets.append(f"Motion reduced reliability ({quality.motion:.0f}/100 motion quality).")
        if quality.face_visibility < 60: bullets.append(f"Face visibility was limited ({quality.face_visibility:.0f}/100).")
        if consistency.reliable_regions < 2: bullets.append(f"Only {consistency.reliable_regions}/3 regions met the reliability gate.")
        if not bullets: bullets.append("The visual and physiological evidence did not support a stable decision.")
        conclusion = "PhysioGuard did not force an authenticity decision because the evidence was insufficient or contradictory."
    return {"title": "Why was this video flagged?" if fusion.verdict == "POTENTIAL DEEPFAKE" else "Why no decision?" if fusion.verdict == "INCONCLUSIVE" else "Why this result?",
            "bullets": bullets, "conclusion": conclusion, "model_status": visual_status}


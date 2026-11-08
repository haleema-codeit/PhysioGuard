"""Cross-region physiological compatibility metrics."""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np

from config import ROI_NAMES
from modules.rppg import RPPGSignal


@dataclass
class PairConsistency:
    left: str
    right: str
    pearson: float
    normalized_similarity: float
    frequency_consistency: float
    score: float

    def as_dict(self):
        return asdict(self)


@dataclass
class ConsistencyResult:
    score: float
    matrix: Dict[str, Dict[str, float]]
    pairs: list
    reliable_regions: int
    physiological_evidence: float
    notes: list

    def as_dict(self):
        return asdict(self)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 8:
        return 0.0
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare(a: RPPGSignal, b: RPPGSignal) -> PairConsistency:
    n = min(a.signal.size, b.signal.size)
    corr = _corr(a.signal[:n], b.signal[:n])
    similarity = float(np.clip((corr + 1.0) * 50.0, 0, 100))
    if a.frequency is None or b.frequency is None:
        freq = 0.0
    else:
        freq = float(np.clip(100.0 - abs(a.frequency - b.frequency) / 1.2 * 100.0, 0, 100))
    score = float(np.clip(0.65 * similarity + 0.35 * freq, 0, 100))
    return PairConsistency(a.region, b.region, corr, similarity, freq, score)


def calculate_consistency(signals: Dict[str, RPPGSignal], region_reliability: Dict[str, float]) -> ConsistencyResult:
    matrix = {a: {b: (100.0 if a == b else 0.0) for b in ROI_NAMES} for a in ROI_NAMES}
    pairs = []
    for i, left in enumerate(ROI_NAMES):
        for right in ROI_NAMES[i + 1:]:
            pair = compare(signals[left], signals[right])
            pairs.append(pair)
            matrix[left][right] = matrix[right][left] = pair.score
    weights = []
    scores = []
    for pair in pairs:
        weight = min(region_reliability.get(pair.left, 0.0), region_reliability.get(pair.right, 0.0)) / 100.0
        weights.append(weight)
        scores.append(pair.score)
    score = float(np.average(scores, weights=weights)) if any(weights) else 0.0
    reliable_regions = sum(1 for name in ROI_NAMES if region_reliability.get(name, 0) >= 45 and signals[name].usable)
    notes = []
    if reliable_regions < 2:
        notes.append("Fewer than two facial regions produced reliable physiological evidence.")
    if score < 45 and reliable_regions >= 2:
        notes.append("Cross-region signals were weakly compatible in the analyzed interval.")
    if score >= 70:
        notes.append("The reliable regions show broadly compatible temporal and frequency behavior.")
    return ConsistencyResult(score, matrix, pairs, reliable_regions, 100.0 - score, notes)


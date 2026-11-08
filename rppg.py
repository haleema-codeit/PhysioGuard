"""POS-style remote photoplethysmography from facial ROI reflectance traces."""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np

from config import PHYSIO_BAND, ROI_NAMES


@dataclass
class RPPGSignal:
    region: str
    time: np.ndarray
    signal: np.ndarray
    frequency: Optional[float]
    bpm: Optional[float]
    quality: float
    usable: bool
    sample_count: int
    spectrum_frequency: np.ndarray
    spectrum_amplitude: np.ndarray
    method: str = "POS-style chrominance projection"

    def as_dict(self) -> dict:
        data = asdict(self)
        for key in ("time", "signal", "spectrum_frequency", "spectrum_amplitude"):
            data[key] = np.asarray(data[key]).tolist()
        return data


def _effective_fps(timestamps: Iterable[float], fallback: float) -> float:
    """Estimate the sampling rate represented by the supplied timestamps.

    The video reader samples frames at a reduced rate and an ROI can have
    additional gaps when face detection fails.  Using the container FPS in
    either case gives the FFT the wrong time scale, so use the median spacing
    between the timestamps that actually produced samples.
    """
    times = np.asarray(list(timestamps), dtype=np.float64)
    if times.size >= 2:
        deltas = np.diff(times)
        deltas = deltas[np.isfinite(deltas) & (deltas > 1e-9)]
        if deltas.size:
            estimated = 1.0 / float(np.median(deltas))
            if np.isfinite(estimated) and estimated > 0:
                return estimated
    return float(fallback) if np.isfinite(fallback) and fallback > 0 else 15.0


def _bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    if signal.size < 8:
        return signal
    # A dependency-free detrending/high-pass + smoothing approximation. scipy,
    # when installed, gives a true Butterworth band-pass.
    try:
        from scipy.signal import butter, filtfilt
        low, high = PHYSIO_BAND
        b, a = butter(3, [low / (fps / 2), high / (fps / 2)], btype="band")
        if len(signal) > 24:
            return filtfilt(b, a, signal)
    except Exception:
        pass
    window = min(signal.size, max(3, int(round(fps * 0.7))))
    if window % 2 == 0:
        window = max(3, window - 1)
    if window >= signal.size:
        return signal
    trend = np.convolve(signal, np.ones(window) / window, mode="same")
    return signal - trend


def pos_projection(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Extract a chrominance signal from frame-wise mean RGB values.

    This uses the POS idea: normalized RGB channels are projected into two
    chrominance axes and combined adaptively over the complete clip.
    """
    if rgb.size == 0:
        return np.empty(0)
    x = np.asarray(rgb, dtype=np.float64)
    x = x / np.maximum(np.mean(x, axis=1, keepdims=True), 1e-8)
    s1 = x[:, 1] - x[:, 2]
    s2 = x[:, 1] + x[:, 2] - 2.0 * x[:, 0]
    scale = np.std(s1) / max(np.std(s2), 1e-8)
    signal = s1 + scale * s2
    return _bandpass(signal, fps)


def _frequency_features(signal: np.ndarray, fps: float):
    if signal.size < max(16, int(fps * 2)):
        return None, None, 0.0
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / fps)
    mask = (freqs >= PHYSIO_BAND[0]) & (freqs <= min(PHYSIO_BAND[1], fps / 2 - 1e-6))
    if not np.any(mask):
        return freqs, spectrum, 0.0
    band_amp = spectrum[mask]
    peak_i = int(np.argmax(band_amp))
    peak = float(freqs[mask][peak_i])
    total = float(np.sum(band_amp))
    prominence = float(band_amp[peak_i] / max(total, 1e-8))
    # A dominant band peak, stable amplitude, and enough duration are needed.
    quality = float(np.clip(100.0 * (0.55 * min(1.0, prominence * 8.0) +
                             0.25 * min(1.0, np.std(signal) / 0.01) +
                             0.20 * min(1.0, signal.size / max(1.0, fps * 20.0))), 0, 100))
    return freqs, spectrum, quality


def extract_rppg(region: str, rgb: np.ndarray, timestamps: Iterable[float], fps: float) -> RPPGSignal:
    timestamps = np.asarray(list(timestamps), dtype=np.float64)
    fps = _effective_fps(timestamps, fps)
    signal = pos_projection(rgb, fps)
    freqs, spectrum, quality = _frequency_features(signal, fps)
    frequency = None
    bpm = None
    if freqs is not None and spectrum is not None and quality >= 25:
        mask = (freqs >= PHYSIO_BAND[0]) & (freqs <= min(PHYSIO_BAND[1], fps / 2 - 1e-6))
        if np.any(mask):
            frequency = float(freqs[mask][np.argmax(spectrum[mask])])
            bpm = frequency * 60.0
    return RPPGSignal(region, timestamps, signal, frequency, bpm, quality,
                      bool(signal.size >= max(16, int(fps * 2)) and quality >= 25),
                      int(signal.size), freqs if freqs is not None else np.empty(0),
                      spectrum if spectrum is not None else np.empty(0))


def extract_multi_region_rppg(frames, observations, timestamps, fps: float) -> Dict[str, RPPGSignal]:
    samples = {name: [] for name in ROI_NAMES}
    valid_times = {name: [] for name in ROI_NAMES}
    for frame, obs, ts in zip(frames, observations, timestamps):
        for name in ROI_NAMES:
            if name not in obs.rois:
                continue
            x, y, w, h = obs.rois[name]
            patch = frame[y:y+h, x:x+w]
            if patch.size == 0:
                continue
            # OpenCV stores BGR; rPPG expects RGB reflectance channels.
            mean_bgr = np.mean(patch.reshape(-1, 3), axis=0)
            samples[name].append(mean_bgr[::-1])
            valid_times[name].append(ts)
    return {name: extract_rppg(name, np.asarray(samples[name]), valid_times[name], fps)
            for name in ROI_NAMES}


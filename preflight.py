"""Runtime dependency and input preflight checks."""

import shutil
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class PreflightStatus:
    python: str
    opencv: str
    ffmpeg: str
    ffprobe: str
    extraction_stack: str
    notes: list

    def as_dict(self):
        return asdict(self)


def check_preflight() -> PreflightStatus:
    import sys

    try:
        import cv2
        opencv = f"available ({cv2.__version__})"
    except Exception as exc:
        opencv = f"unavailable ({type(exc).__name__})"
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    notes = []
    if not ffmpeg_path:
        notes.append("FFmpeg is not installed; OpenCV decoding is used for the current pipeline.")
    if not ffprobe_path:
        notes.append("FFprobe is not installed; metadata is read from OpenCV when available.")
    extraction = "available (OpenCV + NumPy + SciPy)" if opencv.startswith("available") else "unavailable"
    return PreflightStatus(sys.version.split()[0], opencv, ffmpeg_path or "unavailable",
                           ffprobe_path or "unavailable", extraction, notes)


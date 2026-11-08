"""Configuration and shared constants for PhysioGuard."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REAL_DIR = PROJECT_ROOT / "real"
FAKE_DIR = PROJECT_ROOT / "fake"
ANNOTATIONS_PATH = PROJECT_ROOT / "annotations.json"

ROI_NAMES = ("Forehead", "Left Cheek", "Right Cheek")
ROI_COLORS = {
    "Forehead": (34, 211, 238),
    "Left Cheek": (139, 92, 246),
    "Right Cheek": (245, 158, 11),
}
PHYSIO_BAND = (0.7, 4.0)
MIN_ANALYSIS_FRAMES = 60
MAX_ANALYSIS_FRAMES = 360
ANALYSIS_SAMPLE_FPS = 15.0
DEFAULT_FACE_CASCADE = "haarcascade_frontalface_default.xml"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 120.0
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".avi", ".mkv")


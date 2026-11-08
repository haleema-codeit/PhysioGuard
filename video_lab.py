"""Streamlit bridge for the browser-side Video Lab MediaRecorder component."""

from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "physioguard_video_lab",
    path=str(Path(__file__).resolve().parent.parent / "components" / "video_lab"),
)


def video_lab_component(max_bytes: int, supported_extensions: list[str], default=None, mode="all"):
    # The initial height keeps the UI visible while the browser component
    # library is loading and also avoids a zero-height iframe on blocked CDNs.
    return _COMPONENT(height=500, max_bytes=max_bytes, supported_extensions=supported_extensions, mode=mode, default=default)


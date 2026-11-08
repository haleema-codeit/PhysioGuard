"""Premium Streamlit interface for PhysioGuard."""

from __future__ import annotations

import html
import json
import base64
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from evaluation.dataset_loader import discover_dataset, split_known_unseen
from evaluation.evaluate import evaluate
from config import MAX_UPLOAD_BYTES, MAX_VIDEO_DURATION_SECONDS, SUPPORTED_VIDEO_EXTENSIONS
from modules.preflight import check_preflight
from modules.pipeline import analyze_video
from modules.video_lab import video_lab_component

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - requirements include plotly
    go = None


st.set_page_config(page_title="PhysioGuard // AI Video Forensics", page_icon="🛡", layout="wide",
                   initial_sidebar_state="expanded")


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --void:#05070D; --surface:rgba(255,255,255,.045); --line:rgba(255,255,255,.10); --text:#E6EDF7; --muted:#8A97AB; --dim:#5B687D; --cyan:#22D3EE; --violet:#8B5CF6; --emerald:#10B981; --amber:#F59E0B; --crimson:#F43F5E; }
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { color:var(--text); background:radial-gradient(circle at 8% 0%, rgba(34,211,238,.10), transparent 30%), radial-gradient(circle at 90% 100%, rgba(139,92,246,.12), transparent 34%), linear-gradient(135deg,var(--void),#0A1020); }
    [data-testid="stHeader"], footer, #MainMenu { visibility:hidden; height:0; }
    [data-testid="stAppViewContainer"] > .main { padding: 1.2rem 2.2rem 3rem; }
    [data-testid="stSidebar"] { background:rgba(4,7,15,.78); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em; color:var(--text); }
    .mono { font-family:'JetBrains Mono',monospace; }
    .label { font-family:'JetBrains Mono',monospace; color:var(--dim); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; }
    .glass { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; box-shadow:0 12px 34px rgba(0,0,0,.16); backdrop-filter:blur(16px); animation:rise .45s ease both; }
    .glass:hover { border-color:rgba(34,211,238,.28); transform:translateY(-1px); transition:.2s ease; }
    .hero { padding:1.15rem 1.35rem; display:flex; align-items:center; justify-content:space-between; gap:1rem; background:linear-gradient(110deg,rgba(255,255,255,.08),rgba(255,255,255,.025)); border:1px solid rgba(34,211,238,.22); border-radius:20px; box-shadow:0 0 34px rgba(34,211,238,.08); }
    .brand { display:flex; align-items:center; gap:.8rem; }
    .shield { width:42px; height:48px; filter:drop-shadow(0 0 12px rgba(34,211,238,.6)); }
    .wordmark { font:700 1.45rem 'Space Grotesk'; background:linear-gradient(90deg,#E6EDF7,#22D3EE); -webkit-background-clip:text; color:transparent; }
    .submark { color:var(--muted); font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; }
    .chips { display:flex; flex-wrap:wrap; gap:.42rem; justify-content:flex-end; }
    .chip { border:1px solid var(--line); border-radius:999px; color:var(--muted); padding:.35rem .6rem; font: .68rem 'JetBrains Mono'; background:rgba(0,0,0,.14); }
    .status { display:inline-flex; align-items:center; gap:.45rem; color:var(--cyan); font:.72rem 'JetBrains Mono'; letter-spacing:.08em; }
    .dot { width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan); animation:pulse 1.8s infinite; }
    .section { margin:1.55rem 0 .7rem; display:flex; align-items:end; justify-content:space-between; }
    .section h2 { margin:0; font-size:1.25rem; }
    .tag { color:var(--dim); font: .62rem 'JetBrains Mono'; border:1px solid rgba(255,255,255,.09); border-radius:999px; padding:.25rem .48rem; }
    .metric-label { color:var(--muted); font-size:.78rem; }
    .metric-value { color:var(--text); font:600 1.8rem 'Space Grotesk'; margin:.2rem 0; font-variant-numeric:tabular-nums; }
    .metric-note { color:var(--dim); font-size:.7rem; }
    .bar { height:5px; background:rgba(255,255,255,.08); border-radius:999px; overflow:hidden; margin-top:.62rem; }
    .bar > i { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--cyan),var(--violet)); }
    .verdict { border-radius:22px; padding:1.45rem 1.65rem; border:1px solid; position:relative; overflow:hidden; }
    .verdict:after { content:''; position:absolute; inset:0; background:linear-gradient(100deg,transparent,rgba(255,255,255,.08),transparent); transform:translateX(-100%); animation:sweep 4s infinite; }
    .verdict .big { font:700 clamp(2rem,4vw,3.6rem) 'Space Grotesk'; letter-spacing:-.06em; }
    .verdict .desc { color:var(--muted); margin-top:.2rem; }
    .auth { color:var(--emerald); border-color:rgba(16,185,129,.45); box-shadow:0 0 28px rgba(16,185,129,.12); }
    .fake { color:var(--crimson); border-color:rgba(244,63,94,.45); box-shadow:0 0 28px rgba(244,63,94,.12); }
    .uncertain { color:var(--amber); border-color:rgba(245,158,11,.45); box-shadow:0 0 28px rgba(245,158,11,.10); }
    .evidence-row { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.55rem 0; border-bottom:1px solid rgba(255,255,255,.06); }
    .evidence-row:last-child { border-bottom:0; }
    .evidence-row strong { font:600 .88rem 'JetBrains Mono'; color:var(--text); }
    .quality-row { margin:.6rem 0; }
    .quality-top { display:flex; justify-content:space-between; color:var(--muted); font-size:.76rem; }
    .quality-top b { color:var(--text); font-family:'JetBrains Mono'; }
    .chain { display:grid; grid-template-columns:repeat(9,1fr); gap:.25rem; align-items:start; }
    .chain-step { text-align:center; color:var(--muted); font-size:.63rem; }
    .chain-dot { width:22px; height:22px; border-radius:50%; border:1px solid var(--cyan); color:var(--cyan); margin:0 auto .4rem; display:flex; align-items:center; justify-content:center; font: .65rem 'JetBrains Mono'; }
    .small { color:var(--muted); font-size:.8rem; line-height:1.55; }
    .stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid rgba(34,211,238,.34); color:var(--text); background:linear-gradient(90deg,rgba(34,211,238,.15),rgba(139,92,246,.12)); font-weight:600; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan); box-shadow:0 0 18px rgba(34,211,238,.16); }
    [data-testid="stFileUploader"] { border:1px dashed rgba(34,211,238,.35); border-radius:14px; background:rgba(255,255,255,.025); }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk'; }
    /* Frame analysis styles */
    .frame-badge { display:inline-block; padding:.18rem .45rem; border-radius:6px; font:600 .65rem 'JetBrains Mono'; letter-spacing:.06em; }
    .badge-fake  { background:rgba(244,63,94,.22);  color:#F43F5E; border:1px solid rgba(244,63,94,.45); }
    .badge-real  { background:rgba(16,185,129,.18); color:#10B981; border:1px solid rgba(16,185,129,.40); }
    .badge-uncertain { background:rgba(245,158,11,.18); color:#F59E0B; border:1px solid rgba(245,158,11,.40); }
    .badge-failed { background:rgba(91,104,125,.22); color:#8A97AB; border:1px solid rgba(91,104,125,.40); }
    .frame-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.10); border-radius:14px; padding:.55rem; transition:.18s ease; }
    .frame-card:hover { border-color:rgba(34,211,238,.38); background:rgba(34,211,238,.06); transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,.22); }
    .frame-card.selected { border-color:#22D3EE; background:rgba(34,211,238,.10); }
    .frame-meta { font: .62rem 'JetBrains Mono'; color:var(--dim); margin-top:.32rem; }
    .seg-bar { display:flex; height:22px; border-radius:8px; overflow:hidden; width:100%; }
    .seg-real { background:linear-gradient(90deg,#10B981,#059669); }
    .seg-fake { background:linear-gradient(90deg,#F43F5E,#BE123C); }
    .seg-uncertain { background:linear-gradient(90deg,#F59E0B,#D97706); }
    .seg-row { display:flex; align-items:center; gap:.8rem; padding:.5rem 0; border-bottom:1px solid rgba(255,255,255,.06); }
    .seg-row:last-child { border-bottom:0; }
    .timeline-seg { height:28px; display:inline-flex; align-items:center; justify-content:center;
                    font:.58rem 'JetBrains Mono'; color:rgba(255,255,255,.85); transition:.15s ease;
                    cursor:default; }
    .timeline-seg:hover { filter:brightness(1.2); }
    @keyframes pulse { 50% { opacity:.35; transform:scale(.75); } }
    @keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    @keyframes sweep { 60%,100% { transform:translateX(100%); } }
    </style>
    """, unsafe_allow_html=True)


def shield_svg():
    return """<svg class='shield' viewBox='0 0 48 56' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M24 3L43 10V25C43 38 35 48 24 53C13 48 5 38 5 25V10L24 3Z' stroke='#22D3EE' stroke-width='2'/><path d='M15 27L21 33L34 19' stroke='#E6EDF7' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg>"""


def pct(value):
    return f"{float(value):.0f}%"


def _normalize_display_image(image):
    """Convert OpenCV BGR arrays and other image payloads to the RGB form Streamlit expects."""
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image
            return Image.open(io.BytesIO(bytes(image))).convert("RGB")
        except Exception:
            return None
    try:
        arr = np.asarray(image)
    except Exception:
        return image
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
        try:
            import cv2
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            return arr
    return image


def _materialize_upload_file(filename: str, payload: bytes) -> str:
    """Write uploaded bytes to a temp file so Streamlit can render them reliably."""
    suffix = Path(filename).suffix or ".mp4"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(payload)
    finally:
        handle.close()
    return handle.name


def _as_video_preview_path(data, filename: str = "video.mp4", mime_type: str | None = None) -> str | None:
    """Resolve uploaded file bytes or a path into a temp file path Streamlit can display."""
    if data is None:
        return None
    if isinstance(data, (str, Path)):
        path = str(data)
        if Path(path).exists():
            return path
        return None
    if hasattr(data, "getvalue"):
        raw = data.getvalue()
        name = getattr(data, "name", filename) or filename
        return _materialize_upload_file(name, raw)
    if isinstance(data, (bytes, bytearray, memoryview)):
        return _materialize_upload_file(filename, bytes(data))
    return None


def render_image(image):
    if image is None:
        st.info("No image available for this view.")
        return
    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image
            st.image(Image.open(io.BytesIO(bytes(image))).convert("RGB"), use_container_width=True)
            return
        except Exception:
            pass
    arr = _normalize_display_image(image)
    if hasattr(arr, "mode"):
        st.image(arr, use_container_width=True)
        return
    arr = np.asarray(arr)
    if arr.ndim == 2:
        st.image(arr, use_container_width=True)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        st.image(arr[:, :, 0], use_container_width=True)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        st.image(arr[:, :, :3], use_container_width=True)
    else:
        st.image(arr, use_container_width=True)


def section(title, tag="MODEL OUTPUT"):
    st.markdown(f"<div class='section'><div><div class='label'>FORENSIC MODULE</div><h2>{html.escape(title)}</h2></div><span class='tag'>{html.escape(tag)}</span></div>", unsafe_allow_html=True)


def metric_card(label, value, note, bar_value=None):
    bar_value = value if bar_value is None else bar_value
    st.markdown(f"<div class='glass'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(str(value))}</div><div class='metric-note'>{html.escape(note)}</div><div class='bar'><i style='width:{max(0,min(100,float(bar_value))):.1f}%'></i></div></div>", unsafe_allow_html=True)


def plot_signal(signal, title, color, reliability):
    if go is None:
        st.line_chart({title: signal})
        return
    fig = go.Figure(go.Scatter(y=signal, mode="lines", line={"color": color, "width": 1.6},
                               hovertemplate="sample %{x}<br>signal %{y:.4f}<extra></extra>"))
    fig.update_layout(title={"text": f"{title.upper()}  ·  {reliability:.0f}% RELIABLE", "font":{"size":11}},
                      height=175, margin={"l":20,"r":12,"t":32,"b":20}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,.13)", font={"color":"#8A97AB","family":"Inter"},
                      xaxis={"showgrid":False,"title":"sample"}, yaxis={"showgrid":True,"gridcolor":"rgba(255,255,255,.06)"})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_header(result=None):
    if result:
        meta, prov = result["metadata"], result["provenance"]
        case = "PG-" + prov["sha256"][:8].upper()
        status = result["fusion"]["verdict"]
        chips = [meta["filename"], f"{meta['duration']:.1f}s", f"{meta['frames_analyzed']} frames", prov["analysis_timestamp"][:19].replace("T", " ")]
        model = result["visual"]["model_status"]
    else:
        case, status, chips, model = "PG—NO CASE", "STANDBY", ["awaiting upload"], "not initialized"
    chip_html = "".join(f"<span class='chip'>{html.escape(str(c))}</span>" for c in chips)
    st.markdown(f"<div class='hero'><div class='brand'>{shield_svg()}<div><div class='wordmark'>PHYSIOGUARD</div><div class='submark'>AI video forensic analysis</div></div></div><div style='text-align:right'><div class='status'><span class='dot'></span>{html.escape(status)} · <span class='mono'>{case}</span></div><div class='chips'>{chip_html}<span class='chip'>MODEL · {html.escape(model)}</span></div></div></div>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(f"{shield_svg()}<div class='wordmark'>PHYSIOGUARD</div><div class='submark' style='margin-bottom:1.2rem'>Forensic workspace</div>", unsafe_allow_html=True)
        page = st.radio("NAVIGATION", ["Video Lab — Record & Analyze", "Dashboard", "Video Analysis", "Face & Regions", "Physiological Analysis", "Visual Analysis", "Timeline", "Evidence", "Evaluation", "Settings"], index=0, label_visibility="visible")
        st.markdown("<div class='small' style='margin-top:1.5rem'>Physiological consistency is treated as additional evidence. Low-quality or contradictory clips remain inconclusive.</div>", unsafe_allow_html=True)
    return page


def _json_report(result):
    """Return only backend-returned JSON-safe analysis fields."""
    excluded = {"preview_frame", "annotated_frame", "heatmap", "edge_view", "frame_records"}
    data = {key: value for key, value in result.items() if key not in excluded}
    return json.dumps(data, indent=2, default=str)


def _analyze_video_payload(payload):
    """Validate and analyze a browser-selected or MediaRecorder-created blob."""
    if not isinstance(payload, dict) or not payload.get("data"):
        raise ValueError("No video was selected.")
    filename = str(payload.get("filename") or "recording.webm")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {extension or 'unknown'}")
    mime_type = str(payload.get("mime_type") or "")
    if mime_type and not mime_type.startswith("video/"):
        raise ValueError(f"Unsupported MIME type: {mime_type}")
    try:
        raw = base64.b64decode(payload["data"], validate=True)
    except Exception as exc:
        raise ValueError("The selected video data could not be decoded.") from exc
    if not raw:
        raise ValueError("The selected video is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    temp_path = None
    try:
        suffix = extension or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(raw)
            temp_path = handle.name
        import cv2
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            capture.release()
            raise ValueError("The video could not be decoded by the backend.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        duration = frame_count / fps if fps > 0 else float(payload.get("duration") or 0.0)
        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(f"The video is longer than the {MAX_VIDEO_DURATION_SECONDS:.0f}-second limit.")
        return analyze_video(temp_path, original_filename=filename)
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _consume_video_lab_payload(payload):
    if not isinstance(payload, dict):
        return
    if payload.get("type") == "video_selected":
        st.session_state["video_payload"] = payload
    elif payload.get("type") == "cleared":
        st.session_state.pop("video_payload", None)
        st.session_state.pop("analysis", None)


def _render_video_lab_actions():
    selected = st.session_state.get("video_payload")
    if selected:
        raw_data = None
        try:
            raw_data = base64.b64decode(selected["data"], validate=True)
            st.session_state["active_video_bytes"] = raw_data
            st.session_state["active_video_name"] = selected.get("filename", "recording.webm")
            st.session_state["active_video_type"] = selected.get("mime_type", "video/webm")
        except Exception:
            pass

        st.markdown(f"<div class='glass'><div class='label'>SELECTED INPUT</div><h3 style='margin:.35rem 0'> {html.escape(str(selected.get('filename','video')))}</h3><div class='small'>{selected.get('source','video')} · {int(selected.get('size',0))/1024/1024:.2f} MB</div></div>", unsafe_allow_html=True)
        if raw_data:
            preview_path = _as_video_preview_path(raw_data, selected.get("filename", "recording.webm"), selected.get("mime_type", "video/webm"))
            if preview_path:
                st.video(preview_path)

        c1, c2 = st.columns([1, 1])
        with c1:
            analyze_clicked = st.button("ANALYZE VIDEO", type="primary", key="video-lab-analyze")
        with c2:
            reset_clicked = st.button("ANALYZE ANOTHER VIDEO", key="video-lab-reset")
        if reset_clicked:
            st.session_state.pop("video_payload", None)
            st.session_state.pop("analysis", None)
            st.session_state.pop("active_video_bytes", None)
            st.rerun()
        if analyze_clicked:
            try:
                with st.status("Running the synchronous forensic pipeline…", expanded=True) as status:
                    st.write("Validating the uploaded bytes and decoding the video.")
                    st.write("Running face, ROI, rPPG, quality, visual, and fusion analysis.")
                    result = _analyze_video_payload(selected)
                    st.session_state["analysis"] = result
                    status.update(label="Analysis complete", state="complete")
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")
    else:
        st.markdown("<div class='glass'><h3>Choose an input method above</h3><p class='small'>Supported extensions: MP4, WebM, MOV, AVI, MKV · maximum size: 50 MB · maximum duration: 120 seconds.</p></div>", unsafe_allow_html=True)
    if st.session_state.get("analysis"):
        render_result(st.session_state["analysis"])


def render_video_lab():
    section("Video Lab — Record & Analyze", "INPUT / ACTUAL BACKEND")
    st.markdown("<div class='small'>Choose a live camera recording or upload an existing clip. Both paths use the same backend analysis pipeline. Camera permission is requested only after you click Enable Camera.</div>", unsafe_allow_html=True)
    record_tab, upload_tab = st.tabs(["Record Live Video", "Upload Existing Video"])
    with record_tab:
        st.markdown("<div class='label'>LIVE CAPTURE</div><p class='small'>Enable the camera to preview your face, then record a clip. The browser keeps the recording in memory until you use it.</p>", unsafe_allow_html=True)
        payload = video_lab_component(MAX_UPLOAD_BYTES, list(SUPPORTED_VIDEO_EXTENSIONS), default=None, mode="record")
        _consume_video_lab_payload(payload)
    with upload_tab:
        st.markdown("<div class='label'>LOCAL VIDEO</div><p class='small'>Select an MP4, WebM, MOV, AVI, or MKV file. The backend validates actual decodability before analysis.</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Existing Video", type=[x.lstrip('.') for x in SUPPORTED_VIDEO_EXTENSIONS], key="video-lab-upload")
        if uploaded is not None:
            raw = uploaded.getvalue()
            st.session_state["active_video_bytes"] = raw
            st.session_state["active_video_name"] = uploaded.name
            st.session_state["active_video_type"] = uploaded.type or "video/mp4"
            preview_path = _as_video_preview_path(raw, uploaded.name, uploaded.type or "video/mp4")
            if preview_path:
                st.video(preview_path)
            if len(raw) <= MAX_UPLOAD_BYTES:
                _consume_video_lab_payload({"type":"video_selected", "source":"uploaded file", "filename":uploaded.name,
                                            "mime_type":uploaded.type or "video/*", "size":len(raw), "duration":None,
                                            "data":base64.b64encode(raw).decode("ascii")})
            else:
                st.error(f"The selected video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    _render_video_lab_actions()


def render_settings():
    section("Runtime preflight", "ACTUAL ENVIRONMENT")
    status = check_preflight()
    rows = "".join(f"<div class='evidence-row'><span>{html.escape(k.upper())}</span><strong>{html.escape(str(v))}</strong></div>" for k, v in status.as_dict().items() if k != "notes")
    st.markdown(f"<div class='glass'>{rows}</div>", unsafe_allow_html=True)
    for note in status.notes:
        st.warning(note)
    st.info("The current pipeline uses OpenCV decoding directly. FFmpeg and FFprobe are reported when available but are not required for the configured OpenCV path.")


def render_quality(result):
    quality = result["quality"]
    items = [("Face visibility", quality["face_visibility"]), ("Face size", quality["face_size"]),
             ("Lighting", quality["lighting"]), ("Motion quality", quality["motion"]),
             ("Blur/detail", quality["blur"]), ("Compression", quality["compression"]),
             ("Frame stability", quality["frame_stability"]), ("Usable frames", quality["usable_frames"])]
    html_rows = ""
    for name, value in items:
        color = "#10B981" if value >= 65 else "#F59E0B" if value >= 40 else "#F43F5E"
        html_rows += f"<div class='quality-row'><div class='quality-top'><span>{html.escape(name)}</span><b style='color:{color}'>{value:.0f}%</b></div><div class='bar'><i style='width:{value:.1f}%;background:{color}'></i></div></div>"
    gap = result["fusion"].get("quality_normalized_signal_gap", 0.0)
    st.markdown(f"<div class='glass'><div class='label'>DERIVED VISUALIZATION</div><h3 style='margin:.35rem 0 1rem'>VIDEO QUALITY</h3>{html_rows}<div class='small' style='margin-top:.7rem'>Signal reliability <b class='mono'>{quality['signal_reliability']:.0f}/100</b> · video quality <b class='mono'>{quality['video_quality']:.0f}/100</b><br>Quality-normalized signal gap <b class='mono'>{gap:.0f}/100</b></div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Frame badge helpers
# ---------------------------------------------------------------------------

_BADGE_CLASS = {"REAL": "badge-real", "FAKE": "badge-fake",
                "UNCERTAIN": "badge-uncertain", "FAILED": "badge-failed"}
_LABEL_COLOR = {"REAL": "#10B981", "FAKE": "#F43F5E",
                "UNCERTAIN": "#F59E0B", "FAILED": "#5B687D"}
_SEG_COLOR   = {"REAL": "#10B981", "FAKE": "#F43F5E", "UNCERTAIN": "#F59E0B"}


def _badge(label: str) -> str:
    cls = _BADGE_CLASS.get(label, "badge-failed")
    return f"<span class='frame-badge {cls}'>{html.escape(label)}</span>"


def _fmt_ts(ts: float) -> str:
    m = int(ts) // 60
    s = ts - m * 60
    return f"{m}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Frame score distribution chart
# ---------------------------------------------------------------------------

def _render_score_chart(frame_records):
    """Render a Plotly line chart of per-frame fake scores."""
    if go is None:
        return
    classified = [r for r in frame_records if r.fake_score is not None]
    if not classified:
        return
    xs = [r.timestamp for r in classified]
    ys = [r.fake_score for r in classified]
    colors = [_LABEL_COLOR.get(r.label, "#8A97AB") for r in classified]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line={"color": "rgba(139,92,246,.5)", "width": 1.2},
        marker={"color": colors, "size": 5, "line": {"width": 0}},
        hovertemplate="t=%{x:.2f}s<br>fake score=%{y:.1f}<extra></extra>",
        name="Fake score",
    ))
    fig.add_hline(y=60, line_dash="dot", line_color="rgba(244,63,94,.55)",
                  annotation_text="FAKE threshold",
                  annotation_font_color="#F43F5E", annotation_font_size=10)
    fig.add_hline(y=45, line_dash="dot", line_color="rgba(16,185,129,.45)",
                  annotation_text="REAL threshold",
                  annotation_position="bottom right",
                  annotation_font_color="#10B981", annotation_font_size=10)
    fig.add_hrect(y0=45, y1=60, fillcolor="rgba(245,158,11,.07)", line_width=0)
    fig.update_layout(
        height=220, margin={"l": 20, "r": 20, "t": 20, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,.10)",
        font={"color": "#8A97AB", "family": "Inter"},
        xaxis={"title": "timestamp (s)", "showgrid": False},
        yaxis={"title": "fake score (0–100)", "gridcolor": "rgba(255,255,255,.06)",
               "range": [0, 100]},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Temporal segment visualisation
# ---------------------------------------------------------------------------

def _render_segments(segments: list):
    if not segments:
        st.info("No temporal segments were produced (all frames may have failed or the video is too short).")
        return
    total_frames = sum(s["frame_count"] for s in segments)
    if total_frames == 0:
        return
    # Visual timeline bar
    bar_parts = []
    for seg in segments:
        w_pct = seg["frame_count"] / total_frames * 100
        col = _SEG_COLOR.get(seg["label"], "#5B687D")
        label_txt = seg["label"]
        ts_txt = f"{_fmt_ts(seg['start_timestamp'])}→{_fmt_ts(seg['end_timestamp'])}"
        bar_parts.append(
            f"<div class='timeline-seg {seg['label'].lower()}' "
            f"style='width:{w_pct:.1f}%;background:{col};' "
            f"title='{html.escape(label_txt)} {html.escape(ts_txt)}'>"
            f"{label_txt[:1] if w_pct > 4 else ''}</div>"
        )
    bar_html = "<div class='seg-bar'>" + "".join(bar_parts) + "</div>"
    st.markdown(f"<div class='glass'><div class='label'>TEMPORAL TIMELINE</div>{bar_html}</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    for seg in segments:
        col = _SEG_COLOR.get(seg["label"], "#5B687D")
        mean_s = f"{seg['mean_fake_score']:.1f}" if seg["mean_fake_score"] is not None else "—"
        max_s  = f"{seg['max_fake_score']:.1f}"  if seg["max_fake_score"]  is not None else "—"
        st.markdown(
            f"<div class='seg-row'>"
            f"<div style='width:10px;height:10px;border-radius:50%;background:{col};flex-shrink:0'></div>"
            f"<div style='flex:1'>"
            f"  <b style='color:{col}'>{html.escape(seg['label'])}</b>"
            f"  <span class='mono' style='color:var(--muted);font-size:.7rem;margin-left:.5rem'>"
            f"    {_fmt_ts(seg['start_timestamp'])} → {_fmt_ts(seg['end_timestamp'])}"
            f"  </span>"
            f"</div>"
            f"<div class='mono small' style='white-space:nowrap'>"
            f"  {seg['frame_count']} frames · avg {mean_s}/100 · max {max_s}/100"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Frame-level verdict explanation
# ---------------------------------------------------------------------------

def _render_frame_verdict_explanation(result):
    fusion      = result["fusion"]
    vote        = result.get("frame_vote") or {}
    segments    = result.get("frame_segments") or []
    verdict     = fusion["verdict"]

    total   = vote.get("total_frames", 0)
    fake_n  = vote.get("fake_count", 0)
    real_n  = vote.get("real_count", 0)
    unc_n   = vote.get("uncertain_count", 0)
    fail_n  = vote.get("failed_count", 0)
    mean_fs = vote.get("mean_fake_score")

    frame_vote_label = (
        "FAKE" if fake_n > real_n + unc_n else
        "REAL" if real_n > fake_n + unc_n else "MIXED"
    ) if total > 0 else "NONE"

    agreement = (frame_vote_label == "FAKE" and verdict == "POTENTIAL DEEPFAKE") or \
                (frame_vote_label == "REAL" and verdict == "AUTHENTIC")
    disagree  = (frame_vote_label == "FAKE" and verdict == "AUTHENTIC") or \
                (frame_vote_label == "REAL" and verdict == "POTENTIAL DEEPFAKE")

    bullets = []
    if total > 0:
        bullets.append(
            f"{fake_n}/{total} frames ({100*fake_n/total:.0f}%) scored above the FAKE threshold · "
            f"{real_n}/{total} ({100*real_n/total:.0f}%) scored REAL · "
            f"{unc_n} UNCERTAIN · {fail_n} failed."
        )
    if mean_fs is not None:
        bullets.append(f"Mean per-frame fake score: {mean_fs:.1f}/100.")
    if agree_segs := [s for s in segments if s["label"] == "FAKE"]:
        ts_list = ", ".join(
            f"{_fmt_ts(s['start_timestamp'])}–{_fmt_ts(s['end_timestamp'])}" for s in agree_segs
        )
        bullets.append(f"FAKE segments detected at: {ts_list}.")
    if real_segs := [s for s in segments if s["label"] == "REAL"]:
        ts_list = ", ".join(
            f"{_fmt_ts(s['start_timestamp'])}–{_fmt_ts(s['end_timestamp'])}" for s in real_segs
        )
        bullets.append(f"REAL segments detected at: {ts_list}.")
    if disagree:
        bullets.append(
            "⚠ Frame-level majority vote disagrees with the overall verdict. "
            "The overall verdict is produced by a multi-signal fusion that "
            "incorporates physiological evidence and may differ from a purely "
            "pixel-based per-frame score."
        )
    if agreement:
        bullets.append("Frame-level evidence is consistent with the overall verdict.")

    if total == 0:
        conclusion = "No frame-level evidence is available."
    elif disagree:
        conclusion = (
            f"The per-frame analysis labels most frames {frame_vote_label}, which disagrees "
            f"with the overall {verdict} verdict produced by the physiological+visual fusion. "
            "This discrepancy is disclosed here; the overall verdict is not overridden."
        )
    else:
        conclusion = (
            f"The per-frame scores broadly support the overall {verdict} verdict. "
            "Isolated frames may differ due to pose variation, motion blur, or occlusion."
        )
    bullets_html = "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
    st.markdown(
        f"<div class='glass'><ul class='small'>{bullets_html}</ul>"
        f"<p class='small'><b>Conclusion:</b> {html.escape(conclusion)}</p></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Frame summary stat bar
# ---------------------------------------------------------------------------

def _render_frame_summary(vote: dict, metadata: dict):
    total   = vote.get("total_frames", 0)
    fake_n  = vote.get("fake_count", 0)
    real_n  = vote.get("real_count", 0)
    unc_n   = vote.get("uncertain_count", 0)
    fail_n  = vote.get("failed_count", 0)
    sample_fps = metadata.get("analysis_sample_fps", 15.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Extracted frames", str(total), f"@ {sample_fps:.1f} fps sampling", 0)
    with c2: metric_card("FAKE frames", str(fake_n), f"{100*fake_n/max(total,1):.0f}% of total", 100*fake_n/max(total,1))
    with c3: metric_card("REAL frames", str(real_n), f"{100*real_n/max(total,1):.0f}% of total", 100*real_n/max(total,1))
    with c4: metric_card("UNCERTAIN frames", str(unc_n), f"{100*unc_n/max(total,1):.0f}% of total", 100*unc_n/max(total,1))
    with c5: metric_card("Failed", str(fail_n), f"{100*fail_n/max(total,1):.0f}% of total", 0)


# ---------------------------------------------------------------------------
# Per-frame image display (paginated grid)
# ---------------------------------------------------------------------------

def _render_frame_grid(frame_records, key_prefix: str = "fg"):
    """Paginated, filterable grid of every extracted frame."""
    import io
    from PIL import Image  # soft import — cv2 already present

    if not frame_records:
        st.info("No frames were extracted.")
        return

    # --- Controls row ---
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        filter_label = st.selectbox(
            "Filter frames",
            ["ALL", "FAKE", "REAL", "UNCERTAIN", "FAILED"],
            key=f"{key_prefix}_filter",
        )
    with ctrl2:
        cols_per_row = st.select_slider(
            "Columns", options=[2, 3, 4, 5, 6], value=4, key=f"{key_prefix}_cols"
        )
    with ctrl3:
        page_size = st.select_slider(
            "Frames / page", options=[12, 24, 48, 96], value=24, key=f"{key_prefix}_page_size"
        )
    with ctrl4:
        jump_ts = st.number_input(
            "Jump to timestamp (s)", min_value=0.0, value=0.0, step=0.5,
            key=f"{key_prefix}_jump_ts", format="%.1f"
        )

    # Apply filter
    visible = (
        frame_records if filter_label == "ALL"
        else [r for r in frame_records if r.label == filter_label]
    )

    if not visible:
        st.info(f"No {filter_label} frames in this analysis.")
        return

    # Jump-to-timestamp: find the page that contains the nearest frame.
    best_idx = min(range(len(visible)), key=lambda i: abs(visible[i].timestamp - jump_ts))
    default_page = best_idx // page_size
    num_pages = max(1, (len(visible) + page_size - 1) // page_size)

    # Clamp default_page into range.
    default_page = max(0, min(default_page, num_pages - 1))
    if st.session_state.get(f"{key_prefix}_last_filter") != filter_label:
        st.session_state[f"{key_prefix}_last_filter"] = filter_label
        st.session_state[f"{key_prefix}_page"] = default_page

    page_idx = st.session_state.get(f"{key_prefix}_page", default_page)
    page_idx = max(0, min(page_idx, num_pages - 1))

    st.markdown(
        f"<div class='small'>Showing <b class='mono'>{len(visible)}</b> frames"
        f" ({filter_label}) · Page <b class='mono'>{page_idx+1}</b> of "
        f"<b class='mono'>{num_pages}</b></div>",
        unsafe_allow_html=True,
    )

    # Pagination nav
    nav1, nav2, nav3 = st.columns([1, 6, 1])
    with nav1:
        if st.button("◀ Prev", key=f"{key_prefix}_prev", disabled=page_idx == 0):
            st.session_state[f"{key_prefix}_page"] = page_idx - 1
            st.rerun()
    with nav3:
        if st.button("Next ▶", key=f"{key_prefix}_next", disabled=page_idx >= num_pages - 1):
            st.session_state[f"{key_prefix}_page"] = page_idx + 1
            st.rerun()
    with nav2:
        # Page selector
        if num_pages > 1:
            chosen_page = st.slider(
                "Page", 1, num_pages, page_idx + 1,
                key=f"{key_prefix}_slider", label_visibility="collapsed"
            )
            if chosen_page - 1 != page_idx:
                st.session_state[f"{key_prefix}_page"] = chosen_page - 1
                st.rerun()

    start = page_idx * page_size
    page_records = visible[start: start + page_size]

    # Render the grid
    for row_start in range(0, len(page_records), cols_per_row):
        row_recs = page_records[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_widget, rec in zip(cols, row_recs):
            with col_widget:
                lbl_color = _LABEL_COLOR.get(rec.label, "#8A97AB")
                border_color = {
                    "FAKE": "rgba(244,63,94,.55)",
                    "REAL": "rgba(16,185,129,.45)",
                    "UNCERTAIN": "rgba(245,158,11,.40)",
                }.get(rec.label, "rgba(255,255,255,.10)")

                # Prominently display the extracted frame image using a decoded PIL image
                if rec.image_bytes:
                    try:
                        img = Image.open(io.BytesIO(rec.image_bytes)).convert("RGB")
                        st.image(img, use_container_width=True)
                    except Exception:
                        try:
                            img = _normalize_display_image(rec.image_bytes)
                            if img is not None:
                                st.image(img, use_container_width=True)
                            else:
                                st.caption("⚠ Image decode error")
                        except Exception:
                            st.caption("⚠ Image decode error")
                else:
                    st.caption("⚠ No frame image")

                # Frame status badge & timestamp header
                fs_str = f"{rec.fake_score:.1f}/100" if rec.fake_score is not None else "—"
                cf_str = f"{rec.confidence:.0f}%" if rec.confidence is not None else "—"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin:.3rem 0 .15rem;'>"
                    f"{_badge(rec.label)}"
                    f"<span class='frame-meta'>#{rec.frame_index} · {_fmt_ts(rec.timestamp)}</span>"
                    f"</div>"
                    f"<div class='frame-meta' style='margin-bottom:.25rem;'>Fake: {fs_str} · Conf: {cf_str}</div>",
                    unsafe_allow_html=True,
                )

                # Collapsible details for in-depth inspection
                with st.expander("Details", expanded=False):
                    st.caption(
                        f"Frame #{rec.frame_index} · src #{rec.source_frame_index}"
                        f" · t={rec.timestamp:.2f}s · {rec.model_status}"
                    )
                    if rec.status != "ok":
                        st.error(f"Status: {rec.status} — {rec.error_message}")
                    if rec.details:
                        st.json({k: round(float(v), 4) if isinstance(v, float) else v
                                 for k, v in rec.details.items()})


# ---------------------------------------------------------------------------
# Main render_result
# ---------------------------------------------------------------------------

def render_result(result):
    fusion, quality, consistency = result["fusion"], result["quality"], result["consistency"]
    verdict = fusion["verdict"]
    verdict_class = "auth" if verdict == "AUTHENTIC" else "fake" if verdict == "POTENTIAL DEEPFAKE" else "uncertain"
    icon = "✓" if verdict == "AUTHENTIC" else "!" if verdict == "POTENTIAL DEEPFAKE" else "?"
    subtitle = {"AUTHENTIC":"Evidence sufficiently consistent with an authentic video.", "POTENTIAL DEEPFAKE":"Evidence indicates possible manipulation.", "INCONCLUSIVE":"Available evidence is insufficient or contradictory."}[verdict]
    verdict_detail = ("NOT RELIABLE — " + ", ".join(fusion["reasons"]) if verdict == "INCONCLUSIVE" and fusion["reasons"]
                      else f"Quality-aware evidence score: {fusion['final_evidence']:.0f}/100")
    section("Final verdict", "MODEL OUTPUT")
    st.markdown(f"<div class='verdict {verdict_class}'><div class='label'>FORENSIC SYNTHETIC PROBABILITY</div><div class='big'>{icon} {html.escape(verdict)}</div><div class='desc'>{html.escape(subtitle)}</div><div class='small' style='margin-top:.8rem'>{html.escape(verdict_detail)}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Visual manipulation", f"{fusion['visual_manipulation']:.0f}/100", "independent visual module", fusion["visual_manipulation"])
    with c2: metric_card("Physiological consistency", f"{fusion['physiological_consistency']:.0f}/100", "cross-region compatibility", fusion["physiological_consistency"])
    with c3: metric_card("Signal reliability", f"{fusion['signal_reliability']:.0f}/100", "quality-aware rPPG weight", fusion["signal_reliability"])
    with c4: metric_card("Video quality", f"{fusion['video_quality']:.0f}/100", "decoded frame quality", fusion["video_quality"])

    # -----------------------------------------------------------------------
    # Analyzed Video & Facial Tracking
    # -----------------------------------------------------------------------
    section("Analyzed video & facial tracking", "INPUT & STAGED ANALYSIS")
    vid_col1, vid_col2 = st.columns([1, 1])
    with vid_col1:
        st.markdown("<div class='glass'><div class='label'>INPUT VIDEO PLAYER</div>", unsafe_allow_html=True)
        active_bytes = st.session_state.get("active_video_bytes")
        if active_bytes:
            fmt = st.session_state.get("active_video_type", "video/mp4")
            st.video(active_bytes, format=fmt)
        else:
            st.info("Input video stream")
        st.markdown("</div>", unsafe_allow_html=True)
    with vid_col2:
        st.markdown("<div class='glass'><div class='label'>ANALYZED VIDEO · TRACKED FACE · ROIs</div>", unsafe_allow_html=True)
        annotated_video = result.get("annotated_video")
        if annotated_video:
            st.video(annotated_video)
        else:
            render_image(result.get("annotated_frame"))
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Frame Analysis Summary (NEW)
    # -----------------------------------------------------------------------
    section("Frame analysis summary", "PER-FRAME INFERENCE")
    vote = result.get("frame_vote") or {}
    _render_frame_summary(vote, result["metadata"])

    # -----------------------------------------------------------------------
    # Frame score distribution chart (NEW)
    # -----------------------------------------------------------------------
    frame_records = result.get("frame_records") or []
    if frame_records:
        st.markdown("<div class='glass'><div class='label'>PER-FRAME FAKE SCORE OVER TIME</div>",
                    unsafe_allow_html=True)
        _render_score_chart(frame_records)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # All extracted frames — individual inspection (NEW)
    # -----------------------------------------------------------------------
    section("All extracted frames · individual results", "PER-FRAME INFERENCE")
    st.markdown(
        "<div class='small'>Every frame sampled during analysis is shown below. "
        "Click any frame to expand it and inspect the full image, timestamp, "
        "label, and diagnostic scores. Use the filter to focus on FAKE, REAL, "
        "UNCERTAIN, or FAILED frames.</div>",
        unsafe_allow_html=True,
    )
    _render_frame_grid(frame_records, key_prefix="main_fg")

    # -----------------------------------------------------------------------
    # Temporal segmentation (NEW)
    # -----------------------------------------------------------------------
    section("Temporal analysis · real/fake segments", "PER-FRAME INFERENCE")
    _render_segments(result.get("frame_segments") or [])

    # -----------------------------------------------------------------------
    # Why this verdict? (frame-aware explanation, NEW)
    # -----------------------------------------------------------------------
    section("Why this verdict? · frame-level evidence", "MODEL OUTPUT")
    _render_frame_verdict_explanation(result)

    # -----------------------------------------------------------------------
    # Face analysis stage (existing)
    # -----------------------------------------------------------------------
    section("Face analysis stage", "DERIVED VISUALIZATION")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<div class='glass'><div class='label'>ACTUAL ANALYZED FRAME · MESH GUIDES · ROI OVERLAY</div>", unsafe_allow_html=True)
        render_image(result["annotated_frame"])
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        render_quality(result)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        region_html = ""
        for name, value in quality["region_reliability"].items():
            state = "RELIABLE" if value >= 45 else "LOW SIGNAL"
            region_html += f"<div class='evidence-row'><span>{html.escape(name)}</span><strong>{value:.0f}% · {state}</strong></div>"
        st.markdown(f"<div class='glass'><div class='label'>MODEL OUTPUT</div><h3 style='margin:.35rem 0 1rem'>REGION RELIABILITY</h3>{region_html}</div>", unsafe_allow_html=True)

    section("Visual analysis", "MODEL OUTPUT / DERIVED VISUALIZATION")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("<div class='glass'><div class='label'>DERIVED VISUALIZATION</div><h3>Original face</h3>", unsafe_allow_html=True); render_image(result["preview_frame"]); st.markdown("</div>", unsafe_allow_html=True)
    with v2:
        st.markdown("<div class='glass'><div class='label'>FORENSIC EVIDENCE OVERLAY</div><h3>Artifact regions</h3>", unsafe_allow_html=True); render_image(result["heatmap"]); st.markdown("</div>", unsafe_allow_html=True)
    with v3:
        st.markdown(f"<div class='glass'><div class='label'>FORENSIC MAP · {html.escape(result['visual']['model_status'])}</div><h3>Edge / blending evidence</h3>", unsafe_allow_html=True); render_image(result["edge_view"]); st.markdown("</div>", unsafe_allow_html=True)
    regions = result["visual"].get("artifact_regions", [])
    if regions:
        region_text = "".join(
            f"<div class='evidence-row'><span>Region {index}</span><strong>({r['x']}, {r['y']}) · {r['width']}×{r['height']} · anomaly {r['mean_anomaly']:.1f}/100</strong></div>"
            for index, r in enumerate(regions, 1)
        )
        st.markdown(f"<div class='glass'><div class='label'>SPATIAL EVIDENCE</div>{region_text}</div>", unsafe_allow_html=True)
    else:
        st.info("No spatially coherent candidate artifact region was detected inside the analyzed face.")
    st.caption("Candidate regions are produced by a face-constrained pixel-forensic heuristic. The optional classifier provides an image-level score only; it does not provide spatial artifact masks.")

    section("Physiological signal analysis", "MODEL OUTPUT")
    sig_cols = st.columns(3)
    colors = ["#22D3EE", "#8B5CF6", "#F59E0B"]
    for col, (name, signal), color in zip(sig_cols, result["signals"].items(), colors):
        with col:
            with st.container():
                st.markdown(f"<div class='glass'><div class='label'>MODEL OUTPUT</div>", unsafe_allow_html=True)
                bpm = f"{signal['bpm']:.1f} BPM" if signal["bpm"] is not None else "BPM UNAVAILABLE"
                st.markdown(f"<h3 style='margin:.35rem 0 0'>{html.escape(name)}</h3><div class='small'>{bpm} · reliability {signal['quality']:.0f}%</div>", unsafe_allow_html=True)
                plot_signal(signal["signal"], name, color, signal["quality"])
                st.markdown("</div>", unsafe_allow_html=True)

    section("Frequency domain", "DERIVED VISUALIZATION")
    # Use the best region with a real spectrum.  The forehead is not always
    # detected for enough frames, while a cheek may still have valid data.
    available_frequency_signals = [
        (name, signal) for name, signal in result["signals"].items()
        if signal.get("spectrum_frequency") and signal.get("spectrum_amplitude")
    ]
    if available_frequency_signals:
        freq_name, freq_signal = max(
            available_frequency_signals,
            key=lambda item: float(item[1].get("quality", 0.0)),
        )
        if go is not None:
            fig = go.Figure(go.Scatter(x=freq_signal["spectrum_frequency"], y=freq_signal["spectrum_amplitude"],
                                       fill="tozeroy", line={"color":"#22D3EE"}, fillcolor="rgba(34,211,238,.15)"))
            fig.add_vrect(x0=.7, x1=4.0, fillcolor="rgba(139,92,246,.12)", line_width=0)
            fig.update_layout(height=260, margin={"l":20,"r":20,"t":20,"b":30}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,.12)", font={"color":"#8A97AB"}, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        if freq_signal["frequency"] is not None:
            st.markdown(f"<div class='small'>Region: <b class='mono'>{html.escape(freq_name)}</b> · Dominant frequency: <b class='mono'>{freq_signal['frequency']:.3f} Hz</b> · estimated pulse: <b class='mono'>{freq_signal['bpm']:.1f} BPM</b> · reliability: <b class='mono'>{freq_signal['quality']:.0f}%</b></div>", unsafe_allow_html=True)
        else:
            st.info("A spectrum exists, but no dominant frequency passed the current signal-quality gate.")
    else:
        st.info("Frequency-domain representation unavailable: the analyzed sequence is too short or unreliable.")

    section("Cross-region physiological consistency", "MODEL OUTPUT")
    a, b = st.columns([1.35, 1])
    with a:
        matrix = np.array([[consistency["matrix"][r][c] for c in ["Forehead", "Left Cheek", "Right Cheek"]] for r in ["Forehead", "Left Cheek", "Right Cheek"]])
        if go is not None:
            fig = go.Figure(go.Heatmap(z=matrix, x=["Forehead","Left Cheek","Right Cheek"], y=["Forehead","Left Cheek","Right Cheek"], zmin=0, zmax=100, colorscale=[[0,"#24172d"],[.5,"#1f7f9a"],[1,"#22D3EE"]], text=np.round(matrix,0), texttemplate="%{text}", textfont={"family":"JetBrains Mono"}, hovertemplate="%{y} ↔ %{x}<br>%{z:.1f}/100<extra></extra>"))
            fig.update_layout(height=320, margin={"l":10,"r":10,"t":30,"b":20}, paper_bgcolor="rgba(0,0,0,0)", font={"color":"#8A97AB"})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with b:
        metric_card("Overall physiological consistency", f"{consistency['score']:.0f}/100", f"{consistency['reliable_regions']}/3 reliable regions", consistency["score"])
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        for pair in consistency["pairs"]:
            st.markdown(f"<div class='evidence-row'><span>{html.escape(pair['left'])} ↔ {html.escape(pair['right'])}</span><strong>{pair['score']:.0f}/100</strong></div>", unsafe_allow_html=True)

    section("Forensic evidence chain", "MODEL OUTPUT")
    stages = ["VIDEO UPLOADED", "FACE DETECTED", "REGIONS", "rPPG", "QUALITY", "CONSISTENCY", "VISUAL", "FUSION", "VERDICT"]
    chain = "<div class='glass'><div class='chain'>" + "".join(f"<div class='chain-step'><div class='chain-dot'>✓</div>{x}</div>" for x in stages) + "</div></div>"
    st.markdown(chain, unsafe_allow_html=True)

    section(result["explanation"]["title"], "MODEL OUTPUT")
    bullets = "".join(f"<li>{html.escape(x)}</li>" for x in result["explanation"]["bullets"])
    st.markdown(f"<div class='glass'><ul class='small'>{bullets}</ul><p class='small'><b>Conclusion:</b> {html.escape(result['explanation']['conclusion'])}</p></div>", unsafe_allow_html=True)

    section("Provenance / evidence pack", "DERIVED VISUALIZATION")
    prov = result["provenance"]
    st.markdown(f"<div class='glass mono small'>FILE: {html.escape(prov['filename'])}<br>TYPE: {html.escape(prov['file_type'])}<br>SIZE: {prov['file_size_bytes']:,} bytes<br>SHA-256: {prov['sha256']}<br>ANALYSIS: {html.escape(prov['analysis_timestamp'])}<br><span style='color:#F59E0B'>Integrity identifier only — not a claim of originality.</span></div>", unsafe_allow_html=True)
    report = build_report(result)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("EXPORT EVIDENCE PACK · HTML", report, file_name=f"{prov['filename']}.physioguard.html", mime="text/html")
    with d2:
        st.download_button("EXPORT JSON REPORT", _json_report(result), file_name=f"{prov['filename']}.physioguard.json", mime="application/json")


def build_report(result):
    data = {k: v for k, v in result.items() if k not in {"preview_frame", "annotated_frame", "heatmap", "edge_view", "frame_records"}}
    # JSON is intentionally included in a human-readable HTML evidence pack.
    body = html.escape(json.dumps(data, indent=2, default=str))
    return f"<!doctype html><html><head><meta charset='utf-8'><title>PhysioGuard AI-generated forensic analysis report</title><style>body{{font-family:Arial;background:#07101d;color:#e6edf7;max-width:1000px;margin:40px auto;padding:20px}}pre{{white-space:pre-wrap;background:#111b2a;padding:20px;border-radius:12px;border:1px solid #26364d}}h1{{color:#22d3ee}}</style></head><body><h1>PhysioGuard</h1><p>AI-generated forensic analysis report. Not legally certified evidence.</p><pre>{body}</pre></body></html>"


def render_evaluation():
    section("Dataset evaluation", "ACTUAL COMPUTATION")
    st.markdown("<div class='small'>Runs the configured pipeline on the checked-in real/fake videos. No evaluation result is hard-coded. Limit the sample for a live demo.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: max_per_class = st.number_input("Max videos per class", 1, 100, 4)
    with c2:
        sources = sorted({x.dataset_source for x in discover_dataset(".")})
        held_out = st.selectbox("Unseen-source holdout", sources if sources else ["none"])
    if st.button("Run known / unseen evaluation"):
        items = discover_dataset(".", int(max_per_class))
        known, unseen, held = split_known_unseen(items, None if held_out == "none" else held_out)
        with st.spinner(f"Analyzing {len(items)} actual videos…"):
            known_metrics, _ = evaluate(known, ["visual-only", "rppg-only", "visual+rppg", "full-physioguard"])
            unseen_metrics, _ = evaluate(unseen, ["visual-only", "rppg-only", "visual+rppg", "full-physioguard"])
        st.session_state["evaluation"] = {"held_out": held, "known": known_metrics, "unseen": unseen_metrics}
    if st.session_state.get("evaluation"):
        st.json(st.session_state["evaluation"])


def main():
    inject_css()
    page = render_sidebar()
    result = st.session_state.get("analysis")
    render_header(result)
    if page == "Video Lab — Record & Analyze":
        render_video_lab()
        return
    if page == "Settings":
        render_settings()
        return
    if page == "Evaluation":
        render_evaluation(); return
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("UPLOAD FACIAL VIDEO", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="visible")
    if uploaded:
        video_bytes = uploaded.getvalue()
        st.session_state["active_video_bytes"] = video_bytes
        st.session_state["active_video_name"] = uploaded.name
        st.session_state["active_video_type"] = uploaded.type or "video/mp4"
        st.video(video_bytes, format=uploaded.type or "video/mp4")
        if st.button("RUN FORENSIC ANALYSIS", type="primary"):
            suffix = Path(uploaded.name).suffix or ".mp4"
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp.write(video_bytes)
            temp.close()
            with st.spinner("Decoding frames · extracting face regions · estimating signals · fusing evidence…"):
                try:
                    st.session_state["analysis"] = analyze_video(temp.name, original_filename=uploaded.name)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
    if not result:
        section("Ready for analysis", "UNAVAILABLE")
        st.markdown("<div class='glass'><h2>Upload a facial video to open a case.</h2><p class='small'>The system will show the actual analyzed frame, three physiological ROIs, rPPG traces, quality gates, visual evidence, cross-region consistency, temporal markers, and a conservative verdict.</p></div>", unsafe_allow_html=True)
    else:
        render_result(result)


if __name__ == "__main__":
    main()


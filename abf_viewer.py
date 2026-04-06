"""
Interactive ABF File Viewer

A Streamlit app for browsing, visualizing, and exploring patch clamp data
stored in Axon Binary Format (ABF) files. Uses Plotly for interactive
zoom/pan on all traces.

Launch with:
    streamlit run abf_viewer.py
"""

import os
import tempfile
import hashlib
import streamlit as st
import pandas as pd
import pyabf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ABF Viewer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.cdnfonts.com/css/helvetica-neue-55');

    html, body, .stApp,
    .stMarkdown, .stSelectbox, .stMultiSelect,
    .stRadio, .stCheckbox, .stButton > button,
    .stSlider, .stMetric, .stDataFrame,
    input, textarea, select, label, p,
    div:not([data-testid="stIcon"]),
    span:not(.material-symbols-rounded):not(.material-icons),
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Helvetica Neue', 'Helvetica', Arial, sans-serif !important;
    }

    .stApp {
        background-color: #000000;
    }

    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #ffffff;
    }

    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #a1a1aa;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }

    .stDataFrame {
        border: 1px solid #ffffff !important;
        border-radius: 6px;
    }

    div[data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #71717a !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e4e4e7 !important;
    }

    h1, h2, h3 {
        color: #f4f4f5 !important;
    }

    .stPlotlyChart {
        border: 1px solid #ffffff;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#0a0a0a",
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", color="#e4e4e7"),
    hovermode="x unified",
    dragmode="zoom",
    margin=dict(l=60, r=20, t=40, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#222"),
    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#222"),
)

SWEEP_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#6366f1",
]


def color_for_sweep(idx: int) -> str:
    return SWEEP_COLORS[idx % len(SWEEP_COLORS)]


def minmax_downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace to *max_points* using min-max decimation (fully vectorized).

    Each bin contributes its min and max value (in temporal order),
    so spikes and valleys are preserved even after heavy reduction.
    Returns the original arrays unchanged when already small enough.
    """
    n = len(x)
    if n <= max_points:
        return x, y

    n_bins = max_points // 2
    # Trim to an even multiple of n_bins so reshape works cleanly
    usable = n_bins * (n // n_bins)
    y_trim = y[:usable].reshape(n_bins, -1)

    bin_offsets = np.arange(n_bins) * (n // n_bins)
    idx_min = bin_offsets + np.argmin(y_trim, axis=1)
    idx_max = bin_offsets + np.argmax(y_trim, axis=1)

    # Interleave min/max indices, keeping temporal order within each bin
    lo = np.minimum(idx_min, idx_max)
    hi = np.maximum(idx_min, idx_max)
    indices = np.empty(n_bins * 2, dtype=np.intp)
    indices[0::2] = lo
    indices[1::2] = hi

    return x[indices], y[indices]


def _get_temp_dir() -> str:
    """Return a persistent temp directory for the session."""
    if "tmp_dir" not in st.session_state:
        st.session_state["tmp_dir"] = tempfile.mkdtemp(prefix="abf_viewer_")
    return st.session_state["tmp_dir"]


def _save_uploaded(uploaded_file) -> tuple[str, str]:
    """Write an uploaded file to the session temp dir.

    Returns (path, content_hash). Files are deduplicated by content hash
    so re-uploads don't create duplicate copies.
    """
    raw = uploaded_file.getvalue()
    digest = hashlib.md5(raw).hexdigest()
    dest = os.path.join(_get_temp_dir(), f"{digest[:12]}_{uploaded_file.name}")
    if not os.path.exists(dest):
        with open(dest, "wb") as f:
            f.write(raw)
    return dest, digest


@st.cache_data(show_spinner="Loading ABF file…")
def load_abf_metadata(_content_hash: str, path: str) -> dict:
    """Return lightweight metadata without full waveform arrays."""
    abf = pyabf.ABF(path, stimulusFileFolder=os.path.dirname(path))
    channel_info = []
    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        channel_info.append(
            {
                "name": abf.adcNames[ch] if ch < len(abf.adcNames) else f"Ch {ch}",
                "units": abf.adcUnits[ch] if ch < len(abf.adcUnits) else "?",
                "label": abf.sweepLabelY,
            }
        )
    return {
        "filename": os.path.basename(path),
        "path": path,
        "protocol": abf.protocol,
        "sweep_count": abf.sweepCount,
        "channel_count": abf.channelCount,
        "channels": channel_info,
        "sample_rate_hz": abf.dataRate,
        "sweep_length_sec": abf.sweepLengthSec,
        "points_per_sweep": int(abf.sweepLengthSec * abf.dataRate),
        "total_duration_sec": abf.sweepCount * abf.sweepLengthSec,
        "abf_id": abf.abfID,
    }


@st.cache_data(show_spinner="Reading sweep data…")
def load_selected_sweeps(
    _content_hash: str, path: str, channel: int, sweep_indices: tuple[int, ...]
) -> dict:
    """Load only the requested sweep X/Y arrays for a given channel."""
    abf = pyabf.ABF(path, stimulusFileFolder=os.path.dirname(path))
    sweeps = {}
    for i in sweep_indices:
        abf.setSweep(i, channel=channel)
        sweeps[i] = {
            "x": abf.sweepX.copy(),
            "y": abf.sweepY.copy(),
        }
    abf.setSweep(sweep_indices[0], channel=channel)
    return {
        "sweeps": sweeps,
        "x_label": abf.sweepLabelX,
        "y_label": abf.sweepLabelY,
    }


@st.cache_data(show_spinner="Reading command channel…")
def load_selected_commands(
    _content_hash: str, path: str, sweep_indices: tuple[int, ...]
) -> dict:
    """Load sweep command (stimulus) waveform for the requested sweeps."""
    abf = pyabf.ABF(path, stimulusFileFolder=os.path.dirname(path))
    commands = {}
    for i in sweep_indices:
        abf.setSweep(i)
        commands[i] = {
            "x": abf.sweepX.copy(),
            "y": abf.sweepC.copy(),
        }
    abf.setSweep(sweep_indices[0])
    return {
        "commands": commands,
        "x_label": abf.sweepLabelX,
        "y_label": abf.sweepLabelC,
    }


# ── Sidebar: upload & file selection ─────────────────────────────────────────

st.sidebar.title("ABF Viewer")

uploaded_files = st.sidebar.file_uploader(
    "Upload ABF files",
    type=["abf"],
    accept_multiple_files=True,
    help="Drag and drop one or more .abf files, or click to browse.",
    label_visibility="collapsed",
)

max_points = st.sidebar.slider(
    "Max points per trace",
    min_value=1000,
    max_value=50000,
    value=8000,
    step=1000,
    help=(
        "Controls how many points are displayed per sweep. "
        "Lower = faster & smoother; higher = more detail. "
        "Min-max decimation preserves spikes and valleys."
    ),
)

if uploaded_files:
    file_map: dict[str, tuple[str, str]] = {}
    for uf in uploaded_files:
        path, digest = _save_uploaded(uf)
        file_map[uf.name] = (path, digest)
    st.session_state["file_map"] = file_map

file_map: dict[str, tuple[str, str]] = st.session_state.get("file_map", {})

selected_file: str | None = None
content_hash: str | None = None

if file_map:
    names = sorted(file_map.keys())
    st.sidebar.markdown(f"**{len(names)}** ABF file{'s' if len(names) != 1 else ''} loaded")
    chosen_name = st.sidebar.selectbox("Select file", names)
    selected_file, content_hash = file_map[chosen_name]

# ── Sidebar: file details table ──────────────────────────────────────────────

if selected_file is not None:
    meta = load_abf_metadata(content_hash, selected_file)

    st.sidebar.markdown("#### File Details")

    details_df = pd.DataFrame(
        {
            "Property": [
                "ABF ID",
                "Protocol",
                "Sweeps",
                "Channels",
                "Sample Rate",
                "Sweep Length",
                "Points / Sweep",
                "Total Duration",
            ],
            "Value": [
                meta["abf_id"],
                meta["protocol"],
                str(meta["sweep_count"]),
                str(meta["channel_count"]),
                f"{meta['sample_rate_hz'] / 1000:.1f} kHz",
                f"{meta['sweep_length_sec'] * 1000:.1f} ms",
                f"{meta['points_per_sweep']:,}",
                f"{meta['total_duration_sec']:.2f} s ({meta['total_duration_sec'] / 60:.2f} min)",
            ],
        }
    )
    st.sidebar.dataframe(
        details_df,
        hide_index=True,
        use_container_width=True,
    )

    if meta["channel_count"] > 0:
        channel_df = pd.DataFrame(
            [
                {"Channel": i, "Name": ch["name"], "Units": ch["units"]}
                for i, ch in enumerate(meta["channels"])
            ]
        )
        st.sidebar.dataframe(
            channel_df,
            hide_index=True,
            use_container_width=True,
        )

# ── Main area ────────────────────────────────────────────────────────────────

if selected_file is None:
    st.title("ABF File Viewer")
    st.info(
        "Upload one or more ABF files using the sidebar. "
        "You can zoom, pan, and hover on all plots."
    )
    st.stop()

st.title(meta["filename"])

# ── Controls ─────────────────────────────────────────────────────────────────

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

with ctrl_col1:
    channel_options = [
        f"Ch {i}: {ch['name']} ({ch['units']})"
        for i, ch in enumerate(meta["channels"])
    ]
    selected_channel_label = st.selectbox("Channel", channel_options)
    selected_channel = channel_options.index(selected_channel_label)

with ctrl_col2:
    time_unit = st.radio("Time axis", ["ms", "s"], horizontal=True)

with ctrl_col3:
    if meta["sweep_count"] <= 30:
        default_sweeps = list(range(meta["sweep_count"]))
    else:
        default_sweeps = list(range(min(10, meta["sweep_count"])))
    selected_sweeps = st.multiselect(
        "Sweeps to display",
        options=list(range(meta["sweep_count"])),
        default=default_sweeps,
        format_func=lambda s: f"Sweep {s}",
    )

# Quick sweep-selection helpers
helper_col1, helper_col2, helper_col3 = st.columns(3)
with helper_col1:
    if st.button("Select all sweeps"):
        st.session_state["sweep_sel_all"] = True
        st.rerun()
with helper_col2:
    if st.button("Select first sweep only"):
        st.session_state["sweep_sel_first"] = True
        st.rerun()
with helper_col3:
    if st.button("Select every other sweep"):
        st.session_state["sweep_sel_even"] = True
        st.rerun()

if st.session_state.pop("sweep_sel_all", False):
    selected_sweeps = list(range(meta["sweep_count"]))
if st.session_state.pop("sweep_sel_first", False):
    selected_sweeps = [0]
if st.session_state.pop("sweep_sel_even", False):
    selected_sweeps = list(range(0, meta["sweep_count"], 2))

if not selected_sweeps:
    st.warning("Select at least one sweep to display.")
    st.stop()

# ── Load data (only selected sweeps) ─────────────────────────────────────────

sweep_tuple = tuple(sorted(selected_sweeps))
data = load_selected_sweeps(content_hash, selected_file, selected_channel, sweep_tuple)
cmd_data = load_selected_commands(content_hash, selected_file, sweep_tuple)

time_scale = 1000.0 if time_unit == "ms" else 1.0
time_label = "Time (ms)" if time_unit == "ms" else "Time (s)"

# ── Sweep overlay plot ───────────────────────────────────────────────────────

st.subheader("Sweep Overlay")

show_command = st.checkbox("Show command / stimulus channel", value=True)
n_rows = 2 if show_command else 1
row_heights = [0.65, 0.35] if show_command else [1.0]
subplot_titles = [data["y_label"]]
if show_command:
    subplot_titles.append(cmd_data["y_label"])

fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=subplot_titles,
    row_heights=row_heights,
)

for idx in selected_sweeps:
    sweep = data["sweeps"][idx]
    sx, sy = minmax_downsample(sweep["x"], sweep["y"], max_points)
    fig.add_trace(
        go.Scattergl(
            x=sx * time_scale,
            y=sy,
            mode="lines",
            name=f"Sweep {idx}",
            line=dict(color=color_for_sweep(idx), width=1),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )
    if show_command and idx in cmd_data["commands"]:
        cmd = cmd_data["commands"][idx]
        cx, cy = minmax_downsample(cmd["x"], cmd["y"], max_points)
        fig.add_trace(
            go.Scattergl(
                x=cx * time_scale,
                y=cy,
                mode="lines",
                name=f"Cmd {idx}",
                line=dict(color=color_for_sweep(idx), width=1),
                opacity=0.7,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

fig.update_layout(
    height=600 if show_command else 420,
    **PLOTLY_LAYOUT,
)
fig.update_xaxes(title_text=time_label, row=n_rows, col=1)
fig.update_yaxes(title_text=data["y_label"], row=1, col=1)
if show_command:
    fig.update_yaxes(title_text=cmd_data["y_label"], row=2, col=1)

st.plotly_chart(fig, use_container_width=True, key="sweep_overlay")

# ── Average waveform ─────────────────────────────────────────────────────────

st.subheader("Average Waveform")

all_ys = np.array([data["sweeps"][i]["y"] for i in selected_sweeps])
avg_y = np.mean(all_ys, axis=0)
std_y = np.std(all_ys, axis=0)
raw_x = data["sweeps"][selected_sweeps[0]]["x"]

ax, a_avg = minmax_downsample(raw_x, avg_y, max_points)
_, a_upper = minmax_downsample(raw_x, avg_y + std_y, max_points)
_, a_lower = minmax_downsample(raw_x, avg_y - std_y, max_points)

show_std = st.checkbox("Show ± 1 SD envelope", value=True)

fig_avg = go.Figure()
if show_std:
    fig_avg.add_trace(
        go.Scatter(
            x=np.concatenate([ax * time_scale, (ax * time_scale)[::-1]]),
            y=np.concatenate([a_upper, a_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.15)",
            line=dict(width=0),
            name="± 1 SD",
            hoverinfo="skip",
        )
    )
fig_avg.add_trace(
    go.Scattergl(
        x=ax * time_scale,
        y=a_avg,
        mode="lines",
        name=f"Mean (n={len(selected_sweeps)})",
        line=dict(color="#2563eb", width=2),
    )
)
fig_avg.update_layout(
    height=380,
    xaxis_title=time_label,
    yaxis_title=data["y_label"],
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_avg, use_container_width=True, key="avg_waveform")

# ── Continuous (concatenated) time view ──────────────────────────────────────

if meta["sweep_count"] > 1:
    st.subheader("Continuous Time View")
    st.caption(
        "All selected sweeps concatenated end-to-end. "
        "Useful for monitoring slow drift or stability."
    )

    cont_x_parts: list[np.ndarray] = []
    cont_y_parts: list[np.ndarray] = []
    offset = 0.0
    for idx in selected_sweeps:
        sweep = data["sweeps"][idx]
        cont_x_parts.append(sweep["x"] + offset)
        cont_y_parts.append(sweep["y"])
        offset += sweep["x"][-1]

    cont_x_full = np.concatenate(cont_x_parts)
    cont_y_full = np.concatenate(cont_y_parts)

    cx, cy = minmax_downsample(cont_x_full, cont_y_full, max_points)
    cont_time_arr = cx / 60.0

    fig_cont = go.Figure()
    fig_cont.add_trace(
        go.Scattergl(
            x=cont_time_arr,
            y=cy,
            mode="lines",
            line=dict(color="#2563eb", width=1),
            name="Continuous",
        )
    )
    fig_cont.update_layout(
        height=350,
        xaxis_title="Time (min)",
        yaxis_title=data["y_label"],
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_cont, use_container_width=True, key="continuous")

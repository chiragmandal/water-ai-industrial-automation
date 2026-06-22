"""Generate the architecture diagram for the README."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── palette ───────────────────────────────────────────────────────────────────
BG = "#0a0a0f"
SURFACE = "#14141c"
SURFACE2 = "#1c1c28"
SURFACE3 = "#242434"
BORDER = "#2a2a3a"
BORDER_HI = "#3a3a52"
TEXT = "#e4e4ef"
TEXT_DIM = "#8a8aa3"
ACCENT = "#00d4ff"
SUCCESS = "#4ade80"
WARNING = "#fbbf24"
DANGER = "#ef4444"
PURPLE = "#a78bfa"


def box(
    ax,
    x,
    y,
    w,
    h,
    label,
    sublabel="",
    color=SURFACE2,
    border=BORDER_HI,
    text_color=TEXT,
    fontsize=9,
    sublabel_color=TEXT_DIM,
    radius=0.018,
):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.4,
        edgecolor=border,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(
            x + w / 2,
            cy + h * 0.13,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=fontsize,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            x + w / 2,
            cy - h * 0.15,
            sublabel,
            ha="center",
            va="center",
            color=sublabel_color,
            fontsize=fontsize - 1.5,
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            cy,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=fontsize,
            fontweight="bold",
            zorder=4,
        )


def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=BORDER_HI,
    label="",
    lw=1.5,
    style="-|>",
    label_color=TEXT_DIM,
):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw, connectionstyle="arc3,rad=0.0"
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + 0.01,
            my + 0.012,
            label,
            ha="center",
            va="bottom",
            color=label_color,
            fontsize=7,
            style="italic",
            zorder=6,
        )


def section_label(ax, x, y, text, color=TEXT_DIM):
    ax.text(
        x,
        y,
        text.upper(),
        color=color,
        fontsize=6.5,
        fontweight="bold",
        letter_spacing=1.2,
        ha="left",
        va="bottom",
        zorder=6,
    )


# ── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# monkeypatch letter_spacing via fontproperties workaround — just use tracking
# (matplotlib doesn't support letter-spacing natively; we use wider labels)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(
    0.5,
    0.965,
    "Industrial Agent Platform",
    ha="center",
    va="center",
    color=TEXT,
    fontsize=16,
    fontweight="bold",
    zorder=6,
)
ax.text(
    0.5,
    0.942,
    "Architecture Overview",
    ha="center",
    va="center",
    color=TEXT_DIM,
    fontsize=10,
    zorder=6,
)

# glowing accent line under title
ax.plot([0.18, 0.82], [0.932, 0.932], color=ACCENT, lw=1.5, alpha=0.6, zorder=5)

# ── Layer 1: Browser / API Client ─────────────────────────────────────────────
box(
    ax,
    0.30,
    0.855,
    0.40,
    0.058,
    "Browser  /  API Client",
    "http://localhost:8000   ·   curl  ·  MCP client",
    color=SURFACE3,
    border=ACCENT,
    text_color=ACCENT,
    fontsize=9.5,
)

# ── Layer 2: FastAPI ──────────────────────────────────────────────────────────
box(
    ax,
    0.08,
    0.730,
    0.84,
    0.090,
    "FastAPI  +  Uvicorn  (api/main.py)",
    color=SURFACE2,
    border=BORDER_HI,
    fontsize=10,
)

# endpoint badges inside FastAPI box
endpoints = [
    ("GET  /", SUCCESS),
    ("POST /agent/run", ACCENT),
    ("POST /agent/stream  SSE", ACCENT),
    ("POST /sensor/query", WARNING),
    ("POST /anomaly/check", WARNING),
    ("GET  /alerts", PURPLE),
]
badge_xs = [0.10, 0.225, 0.355, 0.525, 0.660, 0.790]
for (ep, col), bx in zip(endpoints, badge_xs):
    badge = FancyBboxPatch(
        (bx, 0.736),
        0.115,
        0.028,
        boxstyle="round,pad=0,rounding_size=0.008",
        linewidth=1,
        edgecolor=col,
        facecolor=SURFACE3,
        zorder=4,
        alpha=0.85,
    )
    ax.add_patch(badge)
    ax.text(
        bx + 0.0575,
        0.750,
        ep,
        ha="center",
        va="center",
        color=col,
        fontsize=6.5,
        fontweight="bold",
        zorder=5,
    )

# arrow: client → fastapi
arrow(ax, 0.50, 0.855, 0.50, 0.820, color=ACCENT, label="HTTP / SSE")

# ── Layer 3: LangGraph Agent ──────────────────────────────────────────────────
box(
    ax,
    0.08,
    0.590,
    0.84,
    0.100,
    "LangGraph ReAct Agent  (agent/graph.py)",
    color=SURFACE2,
    border=BORDER_HI,
    fontsize=10,
)

# Ollama box
box(
    ax,
    0.115,
    0.603,
    0.235,
    0.060,
    "Ollama  (primary)",
    "qwen2.5:7b · local · free",
    color=SURFACE3,
    border=SUCCESS,
    text_color=SUCCESS,
    fontsize=8,
    sublabel_color=TEXT_DIM,
)

# arrow between ollama/azure
ax.text(
    0.385,
    0.634,
    "auto-fallback",
    ha="center",
    va="center",
    color=TEXT_DIM,
    fontsize=7,
    style="italic",
    zorder=6,
)
ax.annotate(
    "",
    xy=(0.420, 0.634),
    xytext=(0.350, 0.634),
    arrowprops=dict(arrowstyle="-|>", color=WARNING, lw=1.2),
    zorder=5,
)
ax.annotate(
    "",
    xy=(0.350, 0.622),
    xytext=(0.420, 0.622),
    arrowprops=dict(arrowstyle="-|>", color=WARNING, lw=1.2),
    zorder=5,
)

# Azure box
box(
    ax,
    0.420,
    0.603,
    0.235,
    0.060,
    "Azure OpenAI  (fallback)",
    "AZURE_OPENAI_* env vars",
    color=SURFACE3,
    border=WARNING,
    text_color=WARNING,
    fontsize=8,
    sublabel_color=TEXT_DIM,
)

# ReAct loop annotation
ax.text(
    0.740,
    0.646,
    "ReAct loop",
    ha="center",
    va="center",
    color=ACCENT,
    fontsize=7.5,
    fontweight="bold",
    zorder=6,
)
loop = mpatches.FancyArrowPatch(
    (0.695, 0.630),
    (0.785, 0.630),
    arrowstyle="<->",
    color=ACCENT,
    lw=1.2,
    connectionstyle="arc3,rad=-0.5",
    zorder=5,
)
ax.add_patch(loop)
ax.text(
    0.740,
    0.609,
    "agent ↔ tools",
    ha="center",
    va="center",
    color=TEXT_DIM,
    fontsize=6.5,
    zorder=6,
)

# arrow: fastapi → agent
arrow(ax, 0.50, 0.730, 0.50, 0.690, color=BORDER_HI, label="invoke / astream")

# ── Layer 4: MCP Tool Layer ───────────────────────────────────────────────────
box(
    ax,
    0.08,
    0.430,
    0.84,
    0.115,
    "MCP Tool Layer  (mcp_server/tools/)",
    color=SURFACE2,
    border=BORDER_HI,
    fontsize=10,
)

tool_defs = [
    (
        "query_sensor_tool",
        "Simulated pump telemetry\nGaussian normal / anomaly\ndistributions · fault injection",
        ACCENT,
    ),
    (
        "check_anomaly_tool",
        "Isolation Forest scoring\nanomaly_score → severity\n(normal / warning / critical)",
        WARNING,
    ),
    (
        "trigger_alert_tool",
        "In-memory alert log\nREST + dashboard panel\nAlert ID · severity · msg",
        DANGER,
    ),
    (
        "list_alerts_tool",
        "Session alert history\npolled every 5 s\nby the dashboard",
        PURPLE,
    ),
]
tw = 0.185
gap = 0.03
start_x = 0.095
for i, (name, desc, col) in enumerate(tool_defs):
    tx = start_x + i * (tw + gap)
    inner = FancyBboxPatch(
        (tx, 0.440),
        tw,
        0.088,
        boxstyle="round,pad=0,rounding_size=0.010",
        linewidth=1.2,
        edgecolor=col,
        facecolor=SURFACE3,
        zorder=4,
    )
    ax.add_patch(inner)
    ax.text(
        tx + tw / 2,
        0.440 + 0.088 - 0.015,
        name,
        ha="center",
        va="top",
        color=col,
        fontsize=7.5,
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        tx + tw / 2,
        0.440 + 0.040,
        desc,
        ha="center",
        va="center",
        color=TEXT_DIM,
        fontsize=6.2,
        zorder=5,
        linespacing=1.4,
    )

# arrow: agent → tools
arrow(ax, 0.50, 0.590, 0.50, 0.545, color=BORDER_HI, label="tool calls")

# ── Layer 5: Backend services ─────────────────────────────────────────────────
# Sensor sim (under query_sensor)
box(
    ax,
    0.095,
    0.295,
    0.185,
    0.090,
    "Sensor Simulator",
    "Gaussian telemetry · 4 pumps\nOPC UA / MQTT in prod",
    color=SURFACE3,
    border=ACCENT,
    text_color=ACCENT,
    fontsize=7.5,
)

# Isolation Forest (under check_anomaly)
box(
    ax,
    0.310,
    0.295,
    0.185,
    0.090,
    "Isolation Forest",
    "Trained on normal samples\nMLflow tracked · joblib artifacts",
    color=SURFACE3,
    border=WARNING,
    text_color=WARNING,
    fontsize=7.5,
)

# Alert log (under trigger_alert)
box(
    ax,
    0.525,
    0.295,
    0.185,
    0.090,
    "Alert Log",
    "In-memory (session)\nReplacement: Redis / CMMS",
    color=SURFACE3,
    border=DANGER,
    text_color=DANGER,
    fontsize=7.5,
)

# MLflow (right side)
box(
    ax,
    0.740,
    0.295,
    0.185,
    0.090,
    "MLflow",
    "Params · metrics · model\nLocal ./mlruns by default",
    color=SURFACE3,
    border=PURPLE,
    text_color=PURPLE,
    fontsize=7.5,
)

# arrows: tools → services
cx_tools = [0.095 + 0.185 / 2, 0.310 + 0.185 / 2, 0.525 + 0.185 / 2]
cx_svc = [0.095 + 0.185 / 2, 0.310 + 0.185 / 2, 0.525 + 0.185 / 2]
for cx in cx_svc:
    arrow(ax, cx, 0.430, cx, 0.385, color=BORDER_HI)

# arrow from Isolation Forest → MLflow
arrow(
    ax,
    0.495,
    0.340,
    0.740,
    0.340,
    color=PURPLE,
    label="logs experiment",
    label_color=PURPLE,
)

# ── MCP stdio server (left side) ──────────────────────────────────────────────
box(
    ax,
    0.08,
    0.145,
    0.240,
    0.100,
    "MCP stdio Server",
    "mcp_server/server.py\nClaude Desktop · Cursor\nany MCP client",
    color=SURFACE3,
    border=BORDER_HI,
    fontsize=8,
)

arrow(ax, 0.200, 0.430, 0.200, 0.245, color=BORDER_HI, label="same tools via stdio")

# ── Azure ML (right side) ─────────────────────────────────────────────────────
box(
    ax,
    0.680,
    0.145,
    0.240,
    0.100,
    "Azure ML (optional)",
    "mlops/azure_ml_pipeline.py\nSame train.py on cloud compute\nStubbed when creds absent",
    color=SURFACE3,
    border=WARNING,
    text_color=WARNING,
    fontsize=8,
)

arrow(ax, 0.760, 0.295, 0.760, 0.245, color=WARNING, label="submit job")

# ── MLflow → Azure ML dashed ──────────────────────────────────────────────────
ax.annotate(
    "",
    xy=(0.800, 0.200),
    xytext=(0.833, 0.295),
    arrowprops=dict(
        arrowstyle="-|>",
        color=WARNING,
        lw=1,
        linestyle="dashed",
        connectionstyle="arc3,rad=0.0",
    ),
    zorder=5,
)

# ── Data flow sequence strip ──────────────────────────────────────────────────
flow_y = 0.075
ax.plot([0.08, 0.92], [flow_y + 0.052, flow_y + 0.052], color=BORDER, lw=0.8, zorder=3)

steps = [
    ("1", "query_sensor_tool", ACCENT),
    ("2", "check_anomaly_tool", WARNING),
    ("3", "trigger_alert_tool", DANGER),
    ("4", "final_answer", SUCCESS),
]
ax.text(
    0.08,
    flow_y + 0.043,
    "Typical agent loop:",
    color=TEXT_DIM,
    fontsize=7,
    va="bottom",
    fontweight="bold",
    zorder=6,
)

step_xs = [0.20, 0.38, 0.56, 0.74]
for i, ((num, label, col), sx) in enumerate(zip(steps, step_xs)):
    circ = plt.Circle((sx, flow_y + 0.020), 0.013, color=col, zorder=5, alpha=0.9)
    ax.add_patch(circ)
    ax.text(
        sx,
        flow_y + 0.020,
        num,
        ha="center",
        va="center",
        color=BG,
        fontsize=8,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        sx,
        flow_y - 0.004,
        label,
        ha="center",
        va="top",
        color=col,
        fontsize=7,
        fontweight="bold",
        zorder=6,
    )
    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(step_xs[i + 1] - 0.018, flow_y + 0.020),
            xytext=(sx + 0.018, flow_y + 0.020),
            arrowprops=dict(arrowstyle="-|>", color=BORDER_HI, lw=1.2),
            zorder=5,
        )

# ── outer border glow ─────────────────────────────────────────────────────────
for alpha, lw in [(0.08, 8), (0.15, 3), (0.35, 1.2)]:
    outer = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ACCENT,
        facecolor="none",
        alpha=alpha,
        zorder=1,
    )
    fig.add_artist(outer)

# ── save ──────────────────────────────────────────────────────────────────────
out = "docs/screenshots/architecture.png"
fig.savefig(
    out,
    dpi=180,
    bbox_inches="tight",
    facecolor=BG,
    edgecolor="none",
    pad_inches=0.18,
)
print(f"Saved → {out}")
plt.close(fig)

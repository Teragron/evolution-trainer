"""Live status dashboard for the evolution-trainer run.

Polls the public `teragron/evo-run` dataset repo -- no token needed -- and
shows current generation, best fitness ever, training speed, and a
best/mean fitness-over-generations chart. See ../README.md (the main repo)
for the training pipeline this watches.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "teragron/evo-run"
REPO_TYPE = "dataset"
PATH_PREFIX = "run"
REFRESH_SECS = 45
GENS_WINDOW = 20  # trailing rows used for the gens/hour estimate

_api = HfApi()

# last-good render, so a transient hub hiccup shows stale data instead of an error
_last_good: Optional[dict] = None


def _download(filename: str) -> str:
    return hf_hub_download(REPO_ID, f"{PATH_PREFIX}/{filename}", repo_type=REPO_TYPE)


def _last_commit_dt() -> Optional[datetime]:
    try:
        info = _api.get_paths_info(REPO_ID, [f"{PATH_PREFIX}/history.csv"],
                                   expand=True, repo_type=REPO_TYPE)
        return info[0].last_commit.date if info else None
    except Exception:                                   # noqa: BLE001
        return None


def _fetch_state() -> dict:
    df = pd.read_csv(_download("history.csv"))
    with open(_download("config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    return {"df": df, "config": config, "commit_dt": _last_commit_dt()}


def _fmt_age(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    seconds = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _make_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["mean"], name="mean",
        line=dict(color="#52514e", width=1.5)))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["best"], name="best",
        line=dict(color="#2a78d6", width=2.5)))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="generation", gridcolor="#e1e0d9"),
        yaxis=dict(title="fitness (m)", gridcolor="#e1e0d9"),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(color="#898781"),
    )
    return fig


def refresh():
    global _last_good
    try:
        state = _fetch_state()
        _last_good = state
        stale = False
    except Exception:                                   # noqa: BLE001
        if _last_good is None:
            empty = go.Figure()
            return "no data yet", "no data yet", "no data yet", "no data yet", empty
        state = _last_good
        stale = True

    df, config = state["df"], state["config"]
    last = df.iloc[-1]
    gen_out = f"{int(last['generation'])}" + (" (stale)" if stale else "")

    best_ever = float(df["best"].max())
    pop = config.get("ga", {}).get("population", "?")
    best_out = f"{best_ever:.2f} m  (pop {pop})"

    window = df.tail(GENS_WINDOW)
    mean_secs = float(window["seconds"].mean()) if len(window) else float("nan")
    gens_per_hour = 3600.0 / mean_secs if mean_secs > 0 else float("nan")
    speed_out = (f"{last['evals_per_sec']:.0f} evals/s, "
                 f"~{gens_per_hour:.0f} gen/hr")

    age_out = _fmt_age(state["commit_dt"]) + (" (fetch failing)" if stale else "")

    return gen_out, best_out, speed_out, age_out, _make_figure(df)


with gr.Blocks(title="Evo Viewer") as demo:
    gr.Markdown("# Evolution training - live status")
    with gr.Row():
        gen_box = gr.Textbox(label="Generation", interactive=False)
        best_box = gr.Textbox(label="Best fitness ever", interactive=False)
        speed_box = gr.Textbox(label="Speed", interactive=False)
        age_box = gr.Textbox(label="Last updated", interactive=False)
    plot_box = gr.Plot(label="Fitness by generation")

    outputs = [gen_box, best_box, speed_box, age_box, plot_box]
    demo.load(fn=refresh, outputs=outputs)
    gr.Timer(REFRESH_SECS).tick(fn=refresh, outputs=outputs)

if __name__ == "__main__":
    demo.launch()

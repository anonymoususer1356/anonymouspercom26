"""Pareto Optimal Selections: every pipeline-stage Pareto frontier, one row of 4.

    python "Plots/pareto_optimal_selections.py"

Standalone: all data for all four panels is embedded below, copied from the
stage-specific scripts (model_selection.py, rag_testing.py, sourcesep_models.py,
speaker_encoder_selection.py) along with the small amount of derivation logic
each panel needs. This script has no dependency on any other file in the repo
except theme.py (shared look-and-feel, not data) -- send both files together
and the recipient can edit the data or the layout freely.

  (a) Embedding combined score vs parameter count  -- from rag_testing.py
  (b) LLM composite score vs parameter count       -- from model_selection.py
  (c) Source-separation quality vs parameter count -- from sourcesep_models.py
  (d) Speaker-encoder EER vs parameter count        -- from speaker_encoder_selection.py

Labels are placed automatically: each starts beside its own point and is then
nudged along y until it clears every other label and stays inside the axes,
with a leader line back to the point it names. Where the scatter is too dense
to also clear every marker, the label's white backing carries legibility.
"""

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import theme  # noqa: E402

# ============================================================================
# Panel (a) data -- LLM shortlist, from model_selection.py's EMBEDDED_DATA.
# model, context_k, intel_idx, omni_idx, aa_lcr, omni_acc, non_hall, hle,
# gpqa, scicode, ifbench. Source: artificialanalysis.ai LLM leaderboard,
# filtered to Size=Tiny, Reasoning=Non-Reasoning.
# ============================================================================
LLM_DATA = [
    ("MiniCPM5-1B (Non-reasoning)", 128.0, 12.0, -1.0, 5.0, 0.0, 99.0, 5.0, 27.0, 1.0, 35.0),
    ("Ministral 3 3B", 256.0, 7.0, -64.0, 16.0, 9.0, 20.0, 5.0, 36.0, 14.0, 27.0),
    ("Phi-4 Mini", 128.0, 6.0, -61.0, 17.0, 10.0, 22.0, 4.0, 33.0, 11.0, 21.0),
    ("Qwen3.5 2B (Non-reasoning)", 262.0, 5.0, -83.0, 15.0, 7.0, 2.0, 5.0, 44.0, 7.0, 29.0),
    ("Granite 4.1 3B", 131.0, 4.0, -77.0, 3.0, 9.0, 5.0, 3.0, 31.0, 12.0, 34.0),
    ("MiniCPM-V 4.6 1.3B", 262.0, 4.0, -83.0, 7.0, 7.0, 4.0, 5.0, 31.0, 2.0, 27.0),
    ("Qwen3.5 0.8B (Non-reasoning)", 262.0, 3.0, -89.0, 9.0, 5.0, 2.0, 5.0, 24.0, 3.0, 22.0),
    ("Exaone 4.0 1.2B (Non-reasoning)", 64.0, 2.0, -82.0, 0.0, 5.0, 8.0, 6.0, 42.0, 7.0, 25.0),
    ("LFM2 2.6B", 32.8, 2.0, -54.0, 0.0, 5.0, 37.0, 5.0, 31.0, 3.0, 20.0),
    ("LFM2.5-1.2B-Instruct", 32.0, 2.0, -72.0, 0.0, 7.0, 15.0, 7.0, 33.0, 2.0, 44.0),
    ("Granite 4.0 H 1B", 128.0, 2.0, -72.0, 6.0, 5.0, 18.0, 5.0, 26.0, 8.0, 26.0),
    ("Gemma 3 270M", 32.0, 2.0, -29.0, 0.0, 1.0, 70.0, 4.0, 22.0, 0.0, 12.0),
    ("Granite 4.0 Micro", 128.0, 2.0, -78.0, 6.0, 9.0, 4.0, 5.0, 34.0, 12.0, 25.0),
    ("Granite 4.0 1B", 128.0, 2.0, -82.0, 6.0, 6.0, 6.0, 5.0, 28.0, 9.0, 21.0),
    ("LFM2.5-VL-1.6B", 32.0, 1.0, -84.0, 0.0, 6.0, 4.0, 5.0, 29.0, 3.0, 33.0),
    ("Granite 4.0 350M", 32.8, 1.0, -69.0, 0.0, 4.0, 24.0, 6.0, 26.0, 1.0, 16.0),
    ("Tiny Aya Global", 8.19, 1.0, -84.0, 0.0, 6.0, 4.0, 5.0, 31.0, 4.0, 20.0),
    ("Granite 4.0 H 350M", 32.8, 1.0, -81.0, 0.0, 4.0, 12.0, 6.0, 26.0, 2.0, 18.0),
]
LLM_DEPLOYED = "Qwen3.5 2B"
LLM_METRICS = ["omni_acc", "gpqa", "aa_lcr", "ifbench"]
LLM_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([BM])\b")

# ============================================================================
# Panel (b) data -- embedding models, from rag_testing.py's EMBEDDED_DATA.
# model, params_M, auc, [recall@1, @5, @10, @20]. Source: run B sweep, 29
# models, AUC over 50,239 permutation pairs, recall@k over 29,368 queries.
# ============================================================================
RAG_DATA = [  # model, params_M, auc, [recall@1, @5, @10, @20]
    ('nomic-embed-text-v1.5', 137, 0.7364, [0.1017, 0.2957, 0.4064, 0.5265]),
    ('bekko-embedding-v1-a25m', 123, 0.7285, [0.0971, 0.2779, 0.3859, 0.5028]),
    ('bekko-embedding-v1-a8m', 106, 0.7106, [0.0913, 0.2622, 0.3666, 0.4783]),
    ('F2LLM-v2-0.6B', 596, 0.7066, [0.099, 0.2759, 0.3798, 0.4941]),
    ('F2LLM-v2-330M', 334, 0.7041, [0.0927, 0.2618, 0.3653, 0.4789]),
    ('e5-base-v2', 110, 0.7034, [0.1058, 0.2987, 0.4082, 0.5271]),
    ('nomic-embed-text-v2-moe', 475, 0.7007, [0.0991, 0.275, 0.3803, 0.4891]),
    ('F2LLM-v2-80M', 80, 0.6916, [0.0793, 0.2248, 0.3164, 0.4219]),
    ('jina-embeddings-v5-text-nano', 212, 0.6858, [0.0988, 0.2813, 0.3871, 0.5007]),
    ('gte-large', 335, 0.6826, [0.0993, 0.2958, 0.4085, 0.5246]),
    ('F2LLM-v2-160M', 159, 0.682, [0.0827, 0.235, 0.3292, 0.4371]),
    ('jina-embeddings-v5-text-small', 596, 0.6746, [0.1002, 0.2807, 0.3833, 0.4911]),
    ('gte-base', 110, 0.6732, [0.0978, 0.2852, 0.396, 0.5072]),
    ('multilingual-e5-small', 118, 0.6702, [0.0882, 0.2484, 0.3457, 0.4537]),
    ('e5-small-v2', 33, 0.6682, [0.101, 0.2771, 0.3785, 0.4848]),
    ('multilingual-e5-large-instruct', 560, 0.666, [0.0986, 0.2754, 0.3787, 0.4875]),
    ('gte-small', 33, 0.6635, [0.0935, 0.2757, 0.382, 0.4892]),
    ('e5-large-v2', 335, 0.6626, [0.1056, 0.2971, 0.4041, 0.5136]),
    ('bge-large-en-v1.5', 335, 0.6598, [0.0994, 0.2829, 0.3897, 0.5025]),
    ('Octen-Embedding-0.6B', 596, 0.6401, [0.0925, 0.2709, 0.3776, 0.4903]),
    ('bge-base-en-v1.5', 110, 0.6395, [0.093, 0.2677, 0.3674, 0.4751]),
    ('mdbr-leaf-mt', 23, 0.6343, [0.0925, 0.2683, 0.369, 0.468]),
    ('bge-small-en-v1.5', 33, 0.6299, [0.0893, 0.2538, 0.3511, 0.4593]),
    ('granite-embedding-97m-r2', 97, 0.602, [0.077, 0.2196, 0.3083, 0.4049]),
    ('bge-m3', 568, 0.5973, [0.0867, 0.2457, 0.3384, 0.4415]),
    ('granite-embedding-311m-r2', 312, 0.5705, [0.0811, 0.2332, 0.3202, 0.4123]),
    ('PIXIE-Rune-v1.0', 568, 0.5685, [0.0851, 0.2504, 0.3454, 0.4446]),
    ('snowflake-arctic-embed-l-v2.0', 568, 0.5239, [0.0749, 0.2171, 0.3007, 0.3969]),
    ('snowflake-arctic-embed-m', 110, 0.4606, [0.0588, 0.164, 0.2328, 0.3146]),
]
RAG_DEPLOYED = "nomic-embed-text-v1.5"
RAG_KS = [1, 5, 10, 20]

# ============================================================================
# Panel (c) data -- source-separation models, from sourcesep_models.py.
# name, params_M, WSJ0-2Mix SI-SNRi/SI-SDRi (dB). Literature comparison, not
# a controlled evaluation of released checkpoints.
# ============================================================================
SOURCESEP_DATA = [
    ("SepTDA", 21.2, 24.0),
    ("SFSRNet", 59.0, 24.0),
    ("MossFormer2", 55.7, 24.1),
    ("QDPN", 200.0, 23.6),
    ("ReSepFormer", 8.0, 18.6),
    ("SPMamba", 6.1, 22.5),
    ("S4M", 3.6, 20.5),
    ("S4M-tiny", 1.8, 19.4),
    ("WA-MISI-5", 32.9, 12.6),
    ("SPN", 56.6, 15.3),
    ("Gated DPRNN", 7.5, 20.1),
    ("SuDoRM-RF (6.4M)", 6.4, 18.9),
    ("pSkiM", 8.5, 15.5),
    ("TF-GridNet", 14.5, 23.5),
    ("SR-CorrNet", 14.0, 24.2),
    ("MossFormer", 42.1, 22.8),
    ("SepEDA", 12.5, 21.2),
    ("TDANet", 2.3, 18.6),
    ("MTDS", 4.0, 21.5),
    ("TFPSNet", 2.7, 21.1),
    ("ReSepNet", 2.8, 21.16),
    ("A-FRCNN", 6.1, 18.3),
    ("Sandglasset", 2.3, 20.8),
    ("SepFormer", 26.0, 22.3),
    ("WaveSplit", 29.0, 22.3),
    ("MSGT-TasNet", 66.8, 17.0),
    ("DPTCN-ATPP", 4.7, 19.6),
    ("DPTNet", 2.7, 20.2),
    ("DPRNN", 2.9, 18.8),
    ("Multi-Decoder DPRNN", 5.2, 20.1),
    ("VSUNOS", 7.5, 20.1),
    ("Two-Step CTN", 8.6, 16.1),
    ("Sudo RM-RF", 2.7, 17.0),
    ("Deep CASA", 12.8, 17.7),
    ("ConvTasNet", 5.1, 15.3),
    ("ADANet", 9.1, 9.1),
    ("TaSNet", 23.6, 13.2),
    ("Chimera++", 32.9, 11.5),
    ("DANet", 9.1, 10.5),
    ("uPIT-BLSTM", 92.7, 9.8),
]

# ============================================================================
# Panel (d) data -- speaker-encoder candidates, from speaker_encoder_selection.py.
# Reported VoxCeleb1-O EER; ranges represented by their midpoint.
# ============================================================================
SPEAKER_ENCODER_DATA = [
    {"name": "ReDimNet-B0", "params_m": 1.0, "eer": 1.07},
    {"name": "ReDimNet2-B0", "params_m": 1.1, "gmacs": 0.33, "eer": 1.04},
    {"name": "NeXt-TDNN-l C128", "params_m": 1.6, "eer": 1.10},
    {"name": "NeXt-TDNN C128", "params_m": 1.9, "eer": 1.03},
    {"name": "ReDimNet-B1", "params_m": 2.2, "eer": 0.73},
    {"name": "ReDimNet2-B1", "params_m": 2.1, "gmacs": 0.56, "eer": 0.78},
    {"name": "ReDimNet-B3", "params_m": 3.0, "eer": 0.47},
    {"name": "ReDimNet2-B2", "params_m": 3.6, "gmacs": 0.95, "eer": 0.57},
    {"name": "Gemini DF-ResNet60", "params_m": 4.1, "eer": 0.94},
    {"name": "ReDimNet2-B3", "params_m": 4.1, "gmacs": 2.70, "eer": 0.42},
    {"name": "DF-ResNet56", "params_m": 4.5, "eer": 0.96},
    {"name": "ReDimNet-B2", "params_m": 4.7, "eer": 0.52},
    {"name": "NeXt-TDNN-l C256", "params_m": 6.0, "eer": 0.81},
    {"name": "ReDimNet-B4", "params_m": 6.3, "eer": 0.44},
    {"name": "ReDimNet2-B4", "params_m": 6.6, "gmacs": 4.62, "eer": 0.37},
    {"name": "ECAPA-TDNN C512", "params_m": 6.4, "eer": 0.94},
    {"name": "TitaNet-S", "params_m": 6.4, "eer": 1.08},
    {"name": "Gemini DF-ResNet114", "params_m": 6.5, "eer": 0.69},
    {"name": "ResNet34", "params_m": 6.65, "eer": 0.895},
    {"name": "ERes2Net", "params_m": 6.6, "eer": 0.84},
    {"name": "NeXt-TDNN C256", "params_m": 7.1, "eer": 0.79},
    {"name": "CAM++", "params_m": 7.18, "eer": 0.72},
    {"name": "LightCAM", "params_m": 8.15, "eer": 0.69},
    {"name": "Gemini DF-ResNet183", "params_m": 9.2, "eer": 0.60},
    {"name": "ReDimNet-B5", "params_m": 9.2, "eer": 0.39},
    {"name": "ReDimNet2-B5", "params_m": 8.9, "gmacs": 9.62, "eer": 0.33},
    {"name": "DF-ResNet233", "params_m": 12.3, "eer": 0.58},
    {"name": "ReDimNet-B6", "params_m": 15.0, "eer": 0.37},
    {"name": "ReDimNet2-B6", "params_m": 12.3, "gmacs": 13.05, "eer": 0.29},
    {"name": "ERes2NetV2", "params_m": 17.8, "eer": 0.61},
    {"name": "TitaNet-L", "params_m": 24.15, "eer": 0.67},
    {"name": "ResNet293", "params_m": 23.8, "eer": 0.53},
    {"name": "ECAPA2", "params_m": 27.1, "eer": 0.39},
    {"name": "Xi-Vector", "params_m": 83.0, "eer": 0.50},
]

# Same frontier/other convention as the individual stage-specific scripts
# (model_selection.py, rag_testing.py, sourcesep_models.py,
# speaker_encoder_selection.py): full teal for the frontier, light purple
# tint for everything else.
FRONTIER_COLOR = theme.GRAYS[5]
OTHER_COLOR = theme.GRAYS[2]
DEPLOYED_COLOR = "#F0544A"  # bright pastel red, distinct from the purple/teal palette
LABEL_FS = 17
TICK_FS = 20
AXIS_FS = 27
CAPTION_FS = 26
OTHER_S = 190
FRONTIER_S = 270
DEPLOYED_S = 560
LEADER = dict(arrowstyle="-", color="#999999", linewidth=0.9, shrinkA=1, shrinkB=8)
GAP = 16


def style_axis(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=AXIS_FS, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS - 3)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS, width=1.8, length=8)
    ax.tick_params(axis="both", which="minor", width=1.2, length=4)
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def pad_ylim(ax, fraction=0.10):
    low, high = ax.get_ylim()
    margin = (high - low) * fraction
    ax.set_ylim(low - margin, high + margin)


def add_labels(ax, items, side="right", bold=()):
    dx = GAP if side == "right" else -GAP
    ha = "left" if side == "right" else "right"
    return [
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, 0),
                    ha=ha, va="center", fontsize=LABEL_FS, arrowprops=LEADER,
                    fontweight="bold" if text in bold else "normal", zorder=10,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.85))
        for text, x, y in items
    ]


def text_box(ann, renderer):
    # Annotation.get_window_extent() unions the text with its leader line,
    # which for a long leader is mostly empty space -- measure the text alone.
    ann.update_positions(renderer)
    return Text.get_window_extent(ann, renderer)


def declutter(fig, panels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_points = 72.0 / fig.dpi

    for ax, anns, markers in panels:
        frame = ax.get_window_extent(renderer)
        dots = [ax.transData.transform(point) for point in markers]
        # A label never crosses its neighbours: rank by anchor height so the
        # stack keeps data order and the relaxation cannot oscillate.
        anchors = [ann.xy[1] for ann in anns]
        rank = {i: r for r, i in enumerate(sorted(range(len(anns)), key=anchors.__getitem__))}

        # Side is decided once, from the measured text width, so a label that
        # cannot fit on its preferred side moves over instead of oscillating.
        for ann in anns:
            width = text_box(ann, renderer).width + GAP / to_points
            anchor_x = ax.transData.transform(ann.xy)[0]
            fits_left = anchor_x - frame.x0 >= width
            fits_right = frame.x1 - anchor_x >= width
            if ann.get_ha() == "right" and not fits_left and fits_right:
                ann.set_ha("left")
                ann.xyann = (GAP, ann.xyann[1])
            elif ann.get_ha() == "left" and not fits_right and fits_left:
                ann.set_ha("right")
                ann.xyann = (-GAP, ann.xyann[1])

        for _ in range(1500):
            boxes = [text_box(a, renderer).expanded(1.04, 1.45) for a in anns]
            push = [0.0] * len(anns)
            slide = [0.0] * len(anns)

            for i in range(len(anns)):
                for j in range(i + 1, len(anns)):
                    if not boxes[i].overlaps(boxes[j]):
                        continue
                    step = (min(boxes[i].y1, boxes[j].y1)
                            - max(boxes[i].y0, boxes[j].y0)) / 2 + 0.5
                    up, down = (i, j) if rank[i] > rank[j] else (j, i)
                    push[up] += step
                    push[down] -= step

            for i, box in enumerate(boxes):
                # A gentle bias off markers only: these scatters are dense
                # enough that demanding full clearance has no solution, and
                # the label's own backing keeps it readable where it lands.
                for x, y in dots:
                    if box.x0 < x < box.x1 and box.y0 < y < box.y1:
                        push[i] += 1.5 if sum(box.intervaly) / 2 >= y else -1.5
                if box.y1 > frame.y1:
                    push[i] -= box.y1 - frame.y1
                if box.y0 < frame.y0:
                    push[i] += frame.y0 - box.y0
                if box.x0 < frame.x0:
                    slide[i] += frame.x0 - box.x0
                if box.x1 > frame.x1:
                    slide[i] -= box.x1 - frame.x1

            if all(abs(p) < 0.5 for p in push) and all(abs(s) < 0.5 for s in slide):
                break
            for ann, shift, drift in zip(anns, push, slide):
                offset_x, offset_y = ann.xyann
                ann.xyann = (offset_x + drift * to_points,
                             offset_y + shift * to_points * 0.8)


def params_billions(model_name):
    """Parse parameter count from the model name (e.g. "Qwen3.5 2B" -> 2.0,
    "Gemma 3 270M" -> 0.27). Returns None if the name carries no explicit
    size."""
    matches = LLM_SIZE_RE.findall(model_name)
    if not matches:
        return None
    value, unit = matches[-1]
    value = float(value)
    return value / 1000 if unit == "M" else value


def panel_llm(ax):
    fields = ["model", "context_k", "intel_idx", "omni_idx", "aa_lcr", "omni_acc",
              "non_hall", "hle", "gpqa", "scicode", "ifbench"]
    rows = [dict(zip(fields, row)) for row in LLM_DATA]
    for r in rows:
        r["model"] = r["model"].replace(" (Non-reasoning)", "")
        r["params_b"] = params_billions(r["model"])

    # Composite score: each of LLM_METRICS min-max scaled, then averaged equally.
    for m in LLM_METRICS:
        vals = [r[m] for r in rows]
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0
        for r in rows:
            r[f"{m}_scaled"] = (r[m] - lo) / span
    for r in rows:
        r["composite"] = sum(r[f"{m}_scaled"] for m in LLM_METRICS) / len(LLM_METRICS)

    sized = [r for r in rows if r["params_b"] is not None]
    # Pareto frontier: fewer params (better) and higher composite (better).
    # Collapse tied sizes to their best composite first.
    best_at_size = {}
    for r in sized:
        current = best_at_size.get(r["params_b"])
        if current is None or r["composite"] > current["composite"]:
            best_at_size[r["params_b"]] = r
    frontier, best = [], -1.0
    for size in sorted(best_at_size):
        r = best_at_size[size]
        if r["composite"] > best:
            frontier.append(r)
            best = r["composite"]
    names = {r["model"] for r in frontier}
    others = [r for r in sized if r["model"] not in names]

    ax.scatter([r["params_b"] for r in others], [r["composite"] for r in others],
               s=OTHER_S, facecolor=OTHER_COLOR, edgecolor="black", linewidth=0.8, zorder=2)
    ax.step([r["params_b"] for r in frontier], [r["composite"] for r in frontier],
            where="post", color="black", linewidth=1.6, zorder=1, linestyle="--")
    ax.scatter([r["params_b"] for r in frontier], [r["composite"] for r in frontier],
               s=FRONTIER_S, facecolor=FRONTIER_COLOR, edgecolor="black",
               linewidth=1.1, zorder=3)

    deployed = next((r for r in sized if r["model"] == LLM_DEPLOYED), None)
    if deployed is not None:
        ax.scatter([deployed["params_b"]], [deployed["composite"]], s=DEPLOYED_S,
                   marker="D", facecolor=DEPLOYED_COLOR, edgecolor="black", linewidth=1.2, zorder=6)

    ax.set_xticks(sorted({r["params_b"] for r in frontier}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    sizes = [r["params_b"] for r in sized]
    ax.set_xlim(min(sizes) * 0.45, max(sizes) * 2.6)
    pad_ylim(ax)
    style_axis(ax, "Billion Parameters", "Composite Score")

    markers = [(r["params_b"], r["composite"]) for r in sized]
    items = [(r["model"], r["params_b"], r["composite"]) for r in frontier]
    return add_labels(ax, items, side="left", bold=(LLM_DEPLOYED,)), markers


def panel_rag(ax):
    rows = [{"model": m, "params_M": params, "auc": auc, "recall": recall}
            for m, params, auc, recall in RAG_DATA]

    # Half AUC, half recall (split evenly across k), each min-max scaled first.
    def scaled(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    auc_s = scaled([r["auc"] for r in rows])
    rec_s = [scaled([r["recall"][i] for r in rows]) for i in range(len(RAG_KS))]
    for j, r in enumerate(rows):
        r["combined"] = (0.5 * auc_s[j]
                         + 0.5 * sum(rec_s[i][j] for i in range(len(RAG_KS))) / len(RAG_KS))

    # Pareto: non-dominated on (fewer params, higher combined score).
    frontier = sorted(
        (r for r in rows if not any(
            o["params_M"] <= r["params_M"] and o["combined"] >= r["combined"]
            and (o["params_M"] < r["params_M"] or o["combined"] > r["combined"])
            for o in rows if o is not r)),
        key=lambda r: r["params_M"])
    names = {r["model"] for r in frontier}
    others = [r for r in rows if r["model"] not in names]

    ax.scatter([r["params_M"] for r in others], [r["combined"] for r in others],
               s=OTHER_S, facecolor=OTHER_COLOR, edgecolor="black", linewidth=0.8, zorder=3)
    ax.step([r["params_M"] for r in frontier], [r["combined"] for r in frontier],
            where="post", color="black", linewidth=1.6, zorder=2)
    ax.scatter([r["params_M"] for r in frontier], [r["combined"] for r in frontier],
               s=FRONTIER_S, facecolor=FRONTIER_COLOR, edgecolor="black",
               linewidth=1.1, zorder=4)

    deployed = next((r for r in rows if r["model"] == RAG_DEPLOYED), None)
    if deployed is not None:
        ax.scatter([deployed["params_M"]], [deployed["combined"]], s=DEPLOYED_S,
                   marker="D", facecolor=DEPLOYED_COLOR, edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_xticks([25, 50, 100, 200, 400, 600])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(6.5, 780)
    pad_ylim(ax)
    style_axis(ax, "Million Parameters", "Combined Score")

    markers = [(r["params_M"], r["combined"]) for r in rows]
    items = [(r["model"], r["params_M"], r["combined"]) for r in frontier]
    return add_labels(ax, items, side="left", bold=(RAG_DEPLOYED,)), markers


def sourcesep_pareto_frontier(rows):
    """Models not dominated by a smaller model with equal or better quality."""
    frontier = []
    for candidate in rows:
        _, parameters, quality = candidate
        dominated = any(
            other is not candidate
            and other[1] <= parameters
            and other[2] >= quality
            and (other[1] < parameters or other[2] > quality)
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: row[1])


def panel_sourcesep(ax):
    frontier = sourcesep_pareto_frontier(SOURCESEP_DATA)
    names = {name for name, _, _ in frontier}
    others = [row for row in SOURCESEP_DATA if row[0] not in names]

    ax.scatter([row[1] for row in others], [row[2] for row in others],
               s=OTHER_S, facecolor=OTHER_COLOR, edgecolor="black", linewidth=0.8, zorder=3)
    ax.step([row[1] for row in frontier], [row[2] for row in frontier],
            where="post", color="black", linewidth=1.6, zorder=2)
    ax.scatter([row[1] for row in frontier], [row[2] for row in frontier],
               s=FRONTIER_S, facecolor=FRONTIER_COLOR, edgecolor="black",
               linewidth=1.1, zorder=4)
    deployed = next(row for row in SOURCESEP_DATA if row[0] == "Sandglasset")
    ax.scatter([deployed[1]], [deployed[2]], s=DEPLOYED_S, marker="D",
               facecolor=DEPLOYED_COLOR, edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_xticks([2, 5, 10, 25, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(0.9, 260)
    pad_ylim(ax)
    style_axis(ax, "Million Parameters", "WSJ0-2Mix SI-SNRi (dB)")

    markers = [(row[1], row[2]) for row in SOURCESEP_DATA]
    ax.text(1.65, 22.0, deployed[0], ha="left", va="bottom", fontsize=LABEL_FS,
            fontweight="bold", zorder=10,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    return [], markers


def speaker_pareto_indices(params, eer):
    """Non-dominated accuracy/size candidates, with both values minimized."""
    frontier = []
    for index in range(len(params)):
        dominated = any(
            other != index
            and params[other] <= params[index]
            and eer[other] <= eer[index]
            and (params[other] < params[index] or eer[other] < eer[index])
            for other in range(len(params))
        )
        if not dominated:
            frontier.append(index)
    return sorted(frontier, key=lambda index: params[index])


def panel_speaker_encoder(ax):
    params = np.asarray([model["params_m"] for model in SPEAKER_ENCODER_DATA])
    eer = np.asarray([model["eer"] for model in SPEAKER_ENCODER_DATA])
    frontier = speaker_pareto_indices(params, eer)
    rest = [i for i in range(len(SPEAKER_ENCODER_DATA)) if i not in frontier]

    ax.scatter(params[rest], eer[rest], s=OTHER_S, facecolor=OTHER_COLOR,
               edgecolor="black", linewidth=0.8, zorder=2)
    ax.step(params[frontier], eer[frontier], where="post", color="black",
            linewidth=1.6, zorder=1)
    ax.scatter(params[frontier], eer[frontier], s=FRONTIER_S,
               facecolor=FRONTIER_COLOR, edgecolor="black", linewidth=1.1, zorder=3)

    deployed = [model for model in SPEAKER_ENCODER_DATA if model["name"] in {"ReDimNet-B1", "ReDimNet2-B6"}]
    ax.scatter([model["params_m"] for model in deployed], [model["eer"] for model in deployed],
               s=DEPLOYED_S, marker="D", facecolor=DEPLOYED_COLOR, edgecolor="black", linewidth=1.2, zorder=4)

    ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(0.85, 115)
    pad_ylim(ax)
    style_axis(ax, "Million Parameters", "VoxCeleb1-O EER (%)")

    markers = list(zip(params, eer))
    items = [(model["name"], params[i], eer[i])
             for i, model in enumerate(SPEAKER_ENCODER_DATA) if model["name"] in {"ReDimNet-B1", "ReDimNet2-B6"}]
    return add_labels(ax, items, side="right", bold=("ReDimNet-B1", "ReDimNet2-B6")), markers


def export_individual() -> None:
    """Each panel as its own standalone figure: no caption/title baked into
    the image (the filename carries that instead), saved to
    Plots/Outputs/Pareto Optimal/<Subcaption>.png."""
    outdir = theme.OUTPUTS / "Pareto Optimal"
    outdir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Embedding Selection", panel_rag),
        ("LLM Selection", panel_llm),
        ("Source-Separation Selection", panel_sourcesep),
        ("Speaker-Encoder Selection", panel_speaker_encoder),
    ]
    legend_handles = [
        Line2D([0], [0], marker="o", color="black", linewidth=1.2, markersize=11,
               markerfacecolor=FRONTIER_COLOR, markeredgecolor="black",
               label="Pareto Frontier"),
        Line2D([0], [0], marker="o", color="none", linestyle="none", markersize=10,
               markerfacecolor=OTHER_COLOR, markeredgecolor="black", label="Other Models"),
        Line2D([0], [0], marker="D", color="none", linestyle="none", markersize=12,
               markerfacecolor=DEPLOYED_COLOR, markeredgecolor="black", markeredgewidth=1.2,
               label="Deployed"),
    ]

    for name, builder in panels:
        theme.apply()
        fig, ax = plt.subplots(figsize=(7.7, 7.7))
        anns, markers = builder(ax)
        ax.legend(handles=legend_handles, loc="best", fontsize=15,
                  frameon=True, facecolor="white", framealpha=0.85,
                  edgecolor="none", handletextpad=0.6, labelspacing=0.7,
                  borderpad=0.6)
        fig.tight_layout()
        declutter(fig, [(ax, anns, markers)])
        fig.savefig(outdir / f"{name}.png", dpi=220,
                    facecolor="white", transparent=False)
        plt.close(fig)
        print(f"  wrote Outputs/Pareto Optimal/{name}.png")


def main() -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 4, figsize=(28.0, 9.2))

    # (a) and (b) swapped: Embedding Selection now leads, LLM Selection second
    # -- (a) has the free lower-left corner the legend now lives in.
    builders = (panel_rag, panel_llm, panel_sourcesep, panel_speaker_encoder)
    panels = [(ax, *builder(ax)) for ax, builder in zip(axes, builders)]

    captions = [
        "(a) Embedding Selection",
        "(b) LLM Selection",
        "(c) Source-Separation Selection",
        "(d) Speaker-Encoder Selection",
    ]
    for ax, caption in zip(axes, captions):
        ax.text(0.5, -0.34, caption, transform=ax.transAxes, ha="center",
                fontsize=CAPTION_FS)

    legend_handles = [
        Line2D([0], [0], marker="o", color="black", linewidth=1.2, markersize=11,
               markerfacecolor=FRONTIER_COLOR, markeredgecolor="black",
               label="Pareto Frontier"),
        Line2D([0], [0], marker="o", color="none", linestyle="none", markersize=10,
               markerfacecolor=OTHER_COLOR, markeredgecolor="black", label="Other Models"),
        Line2D([0], [0], marker="D", color="none", linestyle="none", markersize=12,
               markerfacecolor=DEPLOYED_COLOR, markeredgecolor="black", markeredgewidth=1.2,
               label="Deployed"),
    ]
    # Inside panel (a)'s free lower-left space rather than floating outside
    # the figure -- with no outboard legend to leave room for, the whole row
    # gets to breathe more vertically.
    axes[0].legend(handles=legend_handles, loc="lower left", fontsize=17,
                    frameon=True, facecolor="white", framealpha=0,
                    edgecolor="none", handletextpad=0.6, labelspacing=0.7,
                    borderpad=0.6)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.subplots_adjust(wspace=0.42)
    declutter(fig, panels)
    theme.save(fig, "pareto_optimal_selections")
    plt.close(fig)


if __name__ == "__main__":
    main()
    export_individual()

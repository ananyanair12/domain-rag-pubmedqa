from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "figures" / "13_technical_architecture.png"

NAVY = "#0B2B4F"
TEAL = "#1F8AA8"
GREEN = "#2E9D5B"
AMBER = "#E2A12B"
GREY = "#8A949E"
LIGHT_NAVY = "#EAF1F7"
LIGHT_TEAL = "#DDF1F5"
TEXT = "#17212B"
WHITE = "#FFFFFF"


def rounded_box(ax, xy, width, height, facecolor, edgecolor, radius=0.012, linewidth=1.4):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    return box


def chip(ax, x, y, width, label, status, height=0.055):
    colours = {"Done": GREEN, "Next": AMBER, "Planned": GREY}
    rounded_box(
        ax,
        (x, y),
        width,
        height,
        facecolor=WHITE,
        edgecolor=colours[status],
        linewidth=1.8,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT,
        linespacing=1.05,
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.008, y + height * 0.30),
            0.018,
            height * 0.40,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor=colours[status],
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def draw_layer(ax, y, height, title, chips, fill, header):
    rounded_box(ax, (0.06, y), 0.88, height, fill, NAVY if header != TEAL else TEAL)
    rounded_box(
        ax,
        (0.06, y + height - 0.060),
        0.88,
        0.060,
        header,
        header,
        radius=0.012,
    )
    ax.text(
        0.075,
        y + height - 0.030,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=WHITE,
    )

    for item in chips:
        chip(ax, **item)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # User question arrow entering the interface layer.
    ax.add_patch(
        FancyArrowPatch(
            (0.50, 0.995),
            (0.50, 0.945),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.7,
            color=NAVY,
            clip_on=False,
        )
    )
    ax.text(
        0.515,
        0.975,
        "User question",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )

    draw_layer(
        ax,
        0.795,
        0.145,
        "Interface layer",
        [
            {"x": 0.20, "y": 0.815, "width": 0.26, "label": "Streamlit dashboard", "status": "Planned"},
            {"x": 0.54, "y": 0.815, "width": 0.22, "label": "FastAPI endpoint", "status": "Planned"},
        ],
        WHITE,
        NAVY,
    )

    draw_layer(
        ax,
        0.620,
        0.145,
        "Orchestration layer",
        [
            {"x": 0.16, "y": 0.640, "width": 0.34, "label": "LlamaIndex pipeline (pipeline.py)", "status": "Planned"},
            {"x": 0.54, "y": 0.640, "width": 0.30, "label": "Groq LLM generation (generation.py)", "status": "Planned"},
        ],
        WHITE,
        NAVY,
    )

    draw_layer(
        ax,
        0.445,
        0.145,
        "Retrieval layer",
        [
            {"x": 0.09, "y": 0.465, "width": 0.36, "height": 0.055,
             "label": "Sentence-Transformers baseline\nall-MiniLM-L6-v2", "status": "Done"},
            {"x": 0.40, "y": 0.465, "width": 0.20, "height": 0.055,
             "label": "Fine-tuned embedder", "status": "Next"},
            {"x": 0.635, "y": 0.465, "width": 0.11, "height": 0.055,
             "label": "ChromaDB\nvector search", "status": "Done"},
            {"x": 0.765, "y": 0.465, "width": 0.15, "height": 0.055,
             "label": "Cross-encoder\nreranker", "status": "Planned"},
        ],
        LIGHT_TEAL,
        TEAL,
    )

    draw_layer(
        ax,
        0.270,
        0.145,
        "Knowledge layer",
        [
            {"x": 0.09, "y": 0.290, "width": 0.27, "label": "PubMedQA corpus\n62,249 abstracts", "status": "Done"},
            {"x": 0.38, "y": 0.290, "width": 0.30, "label": "Chunks: 128 / 256 / 512 / whole", "status": "Done"},
            {"x": 0.70, "y": 0.290, "width": 0.20, "label": "ChromaDB index", "status": "Done"},
        ],
        WHITE,
        NAVY,
    )

    draw_layer(
        ax,
        0.095,
        0.145,
        "Evaluation layer",
        [
            {"x": 0.16, "y": 0.115, "width": 0.30, "label": "Hit@k and MRR (eval_pmid.py)", "status": "Done"},
            {"x": 0.50, "y": 0.115, "width": 0.34, "label": "RAGAs (evaluate_ragas.py)", "status": "Planned"},
        ],
        WHITE,
        NAVY,
    )

    # Downward arrows between the five layers.
    arrow_y_pairs = [(0.792, 0.768), (0.617, 0.593), (0.442, 0.418), (0.267, 0.243)]
    for start_y, end_y in arrow_y_pairs:
        ax.add_patch(
            FancyArrowPatch(
                (0.50, start_y),
                (0.50, end_y),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.5,
                color=NAVY,
                clip_on=False,
            )
        )

    legend_y = 0.035
    legend_items = [
        ("Done", GREEN),
        ("Next", AMBER),
        ("Planned", GREY),
    ]
    for index, (label, colour) in enumerate(legend_items):
        x = 0.30 + index * 0.14
        ax.add_patch(
            FancyBboxPatch(
                (x, legend_y),
                0.018,
                0.018,
                boxstyle="round,pad=0.002,rounding_size=0.005",
                facecolor=colour,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(
            x + 0.025,
            legend_y + 0.009,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            color=TEXT,
        )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor=WHITE, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()

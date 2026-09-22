"""Plot measured packaged-model latency; requires the plots extra."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data["results"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#f6f7fb")
    ax.set_facecolor("#f6f7fb")
    ax.plot(
        [r["agents"] for r in rows],
        [r["median_ms"] for r in rows],
        color="#235a78",
        marker="o",
        linewidth=2.5,
    )
    for row in rows:
        ax.scatter(
            [row["agents"]] * len(row["latency_ms"]),
            row["latency_ms"],
            color="#235a78",
            alpha=0.2,
            s=15,
        )
    ax.set(
        xlabel="Independent recurrent agents",
        ylabel="Forward latency (ms)",
        title="SAMS • packaged-model inference scaling",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.12,
        0.025,
        f"{data['device'].upper()} · batch {data['batch_size']} · {data['sequence_length']} tokens · random weights / synthetic inputs\nLine: median; dots: individual runs. Timing is not market-fidelity evidence.",
        fontsize=9,
        color="#52606d",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

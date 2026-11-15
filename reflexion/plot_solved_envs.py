"""
This snippet plots the trial number against the proportion of solved environments
in the Reflexion framework. For reference, see Figure 3a of
title={Reflexion: Language Agents with Verbal Reinforcement Learning}, author={Noah Shinn and Federico Cassano 
and Edward Berman and Ashwin Gopinath and Karthik Narasimhan and Shunyu Yao},
year={2023},
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RECORDS = Path(r"C:\Users\white\Desktop\MSc_thesis\reflexion\playpen-records")
IMG = Path(r"C:\Users\white\Desktop\MSc_thesis\reflexion\selected_episodes\images")
OUT_STEM = "fig_pass_at_k"

MAX_TRIALS = 4
STATIC = {"bbh", "cladder", "ifeval", "mmlu_pro", "eqbench"}

MODELS = {
    "Qwen3.5-4B": "reflexion-Qwen3.5-4B",
    "Qwen3.5-9B": "reflexion-Qwen3.5-9B",
    "Qwen3.5-27B": "reflexion-Qwen3.5-27B",
    "Llama-3.1-8B-Instruct": "reflexion-llama-3.1-8b-instruct",
}

MODEL_COLORS = {"Qwen3.5-4B": "#9ecae1", "Qwen3.5-9B": "#4292c6",
                "Qwen3.5-27B": "#08519c", "Llama-3.1-8B-Instruct": "#d95f02"}
MODEL_MARKERS = {"Qwen3.5-4B": "o", "Qwen3.5-9B": "o",
                "Qwen3.5-27B": "o", "Llama-3.1-8B-Instruct": "o"}


def collect_curve(model_dir, group):
    """Return (cumulative percentages, n_instances) for one model"""
    solved_at = []
    instances = set()

    for trace_path in model_dir.glob("**/reflexion_trace.json"):
        game = trace_path.parent.parent.parent.name
        is_static = game in STATIC
        if group == "games" and is_static:
            continue
        if group == "static" and not is_static:
            continue

        payload = json.load(open(trace_path, encoding="utf-8"))
        experiment = trace_path.parent.parent.name
        gid = int(payload["key"].split("/")[-1])
        instances.add((game, experiment, gid))

        if payload["outcome"] == "Outcome.SUCCESS":
            solved_at.append(payload["trial"])

    n = len(instances)
    cum = []
    for k in range(MAX_TRIALS):
        hits = 0
        for trial in solved_at:
            if trial <= k:
                hits += 1
        cum.append(100 * hits / n)
    return cum, n


def plot(curves, counts, out_stem, group):
    xs = list(range(MAX_TRIALS))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    for model in MODELS:
        if model not in curves:
            continue
        ax.plot(xs, curves[model], marker=MODEL_MARKERS[model], markersize=6,
                linewidth=2, color=MODEL_COLORS[model], label=model,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2)

    ax.set_xticks(xs)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Proportion of Solved Instances")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", bbox_inches="tight", dpi=300)
    print("wrote %s.pdf and %s.png" % (out_stem, out_stem))


def main():
    IMG.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
    })

    for group in ("games", "static"):
        curves = {}
        counts = {}
        for model, folder in MODELS.items():
            model_dir = RECORDS / folder
            cum, n = collect_curve(model_dir, group)
            curves[model] = cum
            counts[model] = n

        print()
        print(group)
        print(("model", "n", "trial0", "<=1", "<=2", "<=3"))
        for model in MODELS:
            if model not in curves:
                continue
            cum = curves[model]
            print((model, counts[model], cum[0], cum[1], cum[2], cum[3]))

        stem = str(IMG / OUT_STEM)
        if group == "static":
            stem += "_static"
        plot(curves, counts, stem, group)


if __name__ == "__main__":
    main()
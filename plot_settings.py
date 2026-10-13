import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#get results file and the setting to highlight
parser = argparse.ArgumentParser()
parser.add_argument("--summary", default="results/summary.csv")
parser.add_argument("--per_image", default="results/per_image.csv")
parser.add_argument("--output", default="results/settings_grid.png")
parser.add_argument("--clip", type=int, default=2)
parser.add_argument("--tile", type=int, default=8)
args = parser.parse_args()

rows = {r["setting"]: r for r in csv.DictReader(open(args.summary))}

panels = [
    ("washout_pct", "Washout (% of tooth pixels at 255)", "lower is better"),
    ("near_white_pct", "Near white (% of tooth pixels at 250+)", "lower is better"),
    ("edge_contrast", "Edge contrast (tooth edge minus bone, gray levels)", "higher is better"),
    ("grain", "Grain inside teeth (fine-detail std, gray levels)", "lower is better"),
]
colors = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a"}
markers = {4: "o", 8: "s", 16: "^"}
clips = [1, 2, 3, 4]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (key, title, note) in zip(axes.flat, panels):
    #one line per tile size across the clip values
    ends = {}
    for t in [4, 8, 16]:
        ys = [float(rows[f"clip{c}_tile{t}"][key]) for c in clips]
        ax.plot(clips, ys, color=colors[t], marker=markers[t], markersize=7, linewidth=2, label=f"tile {t}")
        ends[t] = ys[-1]

    #reference lines for no enhancement and global equalization
    ax.axhline(float(rows["original"][key]), color="#52514e", linestyle=":", linewidth=1.2, label="original (no enhancement)")
    ax.axhline(float(rows["global_eq"][key]), color="#52514e", linestyle="--", linewidth=1.2, label="global equalization")

    #label line ends, nudged apart so they dont overlap
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.06
    last = None
    for t, y in sorted(ends.items(), key=lambda x: x[1]):
        if last is not None and y - last < gap:
            y = last + gap
        ax.annotate(f"tile {t}", (4, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color="#52514e")
        last = y
    #make room if a nudged label went past the top
    if last > hi - gap:
        ax.set_ylim(lo, last + gap)

    #ring around the recommended setting
    y = float(rows[f"clip{args.clip}_tile{args.tile}"][key])
    ax.scatter([args.clip], [y], s=260, facecolors="none", edgecolors="#0b0b0b", linewidths=1.5, zorder=5)

    ax.set_title(f"{title}, {note}", fontsize=10, loc="left")
    ax.set_xticks(clips)
    ax.set_xlim(0.8, 4.7)
    ax.set_xlabel("clipLimit", fontsize=9, color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9, colors="#52514e")

#one shared legend under the title
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=5, frameon=False, fontsize=9)
#count how many images went into the averages
n_images = len(set(r["image"] for r in csv.DictReader(open(args.per_image))))
fig.suptitle(f"CLAHE settings grid over {n_images} labeled X-rays (ring = clip {args.clip}, tile {args.tile})", fontsize=12, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(args.output, dpi=200)
print("saved", args.output)

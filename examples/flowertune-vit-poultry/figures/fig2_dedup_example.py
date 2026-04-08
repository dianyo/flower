"""
Figure 2 image set only: high-res Zenodo (kept) → 3 Roboflow duplicates (removed, red X).

For use below your TikZ flow diagram. Reads logs/deduplication_report_*.json,
picks a random cross-source group, and outputs a single combined image:
  [Zenodo high-res] → [dupe1 ✗] [dupe2 ✗] [dupe3 ✗]

Usage:
  python fig2_dedup_example.py
  python fig2_dedup_example.py --logs-dir /path/to/logs --raw-data /path/to/raw_data
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOWERTUNE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = FLOWERTUNE_ROOT / "logs"
DEFAULT_RAW_DATA = FLOWERTUNE_ROOT / "raw_data"


def find_latest_report(logs_dir: Path):
    if not logs_dir.exists():
        return None
    reports = list(logs_dir.glob("deduplication_report_*.json"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def load_duplicate_groups(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    return data.get("duplicate_groups", [])


def pick_cross_source_group(groups: list):
    """Pick the cross-source group with the most duplicates (prefer Roboflow in removed)."""
    cross = [g for g in groups if g.get("cross_source") and len(g.get("removed", [])) >= 1]
    if not cross:
        return None
    roboflow_in_removed = [
        g for g in cross
        if any(r.get("source", "").startswith("roboflow") for r in g["removed"])
    ]
    pool = roboflow_in_removed if roboflow_in_removed else cross
    # Choose the group with the most duplicates (so we get many dupes for the figure)
    return max(pool, key=lambda g: g.get("num_duplicates", len(g.get("removed", []))))


def draw_red_x(ax, linewidth=4, color="red"):
    """Overlay a red X on the axes (denotes removed)."""
    ax.autoscale(False)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.plot([xmin, xmax], [ymax, ymin], color=color, linewidth=linewidth, solid_capstyle="round")
    ax.plot([xmin, xmax], [ymin, ymax], color=color, linewidth=linewidth, solid_capstyle="round")


def filename_only(path_str: str) -> str:
    """Return only the filename (e.g. salmo.1702.jpg) to avoid overlap in figure."""
    return path_str.replace("\\", "/").split("/")[-1]


def main():
    ap = argparse.ArgumentParser(description="Fig2 image set: Zenodo → 3 Roboflow duplicates (red X).")
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    ap.add_argument("--raw-data", type=Path, default=DEFAULT_RAW_DATA)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "fig2_dedup_example.png")
    args = ap.parse_args()

    json_path = find_latest_report(args.logs_dir)
    if not json_path:
        print("No deduplication_report_*.json found in", args.logs_dir)
        return

    groups = load_duplicate_groups(json_path)
    group = pick_cross_source_group(groups)
    if not group:
        print("No cross-source duplicate group found.")
        return

    kept = group["kept"]
    removed = group["removed"]
    roboflow_removed = [r for r in removed if r.get("source", "").startswith("roboflow")]
    others_removed = [r for r in removed if r not in roboflow_removed]
    selected_removed = (roboflow_removed[:3] if len(roboflow_removed) >= 3 else roboflow_removed) or others_removed[:3]

    kept_path = args.raw_data / kept["path"]
    if not kept_path.exists():
        print("Kept image not found:", kept_path)
        return

    removed_paths = []
    removed_meta = []  # keep path info for labels
    for r in selected_removed:
        p = args.raw_data / r["path"]
        if p.exists():
            removed_paths.append(p)
            removed_meta.append(r)
    if len(removed_paths) < 1:
        print("No removed image paths found under --raw-data.")
        return

    # Layout: [Zenodo] [dupe1] [dupe2] [dupe3] — no arrow, no (a)(b)(c)(d), no titles
    n_removed = min(3, len(removed_paths))
    n_cols = 1 + n_removed  # kept + dupes
    fig, _ = plt.subplots(1, 1, figsize=(2.5 + 1.4 * n_removed, 3.0))
    fig.clear()
    gs = GridSpec(1, n_cols, figure=fig, width_ratios=[1.8] + [1] * n_removed)
    fig.subplots_adjust(wspace=0.1, left=0.02, right=0.98, top=0.92, bottom=0.2)

    # Left: Zenodo (kept) — filename only below
    ax0 = fig.add_subplot(gs[0, 0])
    img_kept = Image.open(kept_path).convert("RGB")
    ax0.imshow(img_kept)
    ax0.axis("off")
    kept_label = filename_only(kept["path"])
    ax0.text(0.5, -0.08, kept_label, transform=ax0.transAxes, ha="center", fontsize=8, family="monospace")

    # Right: removed with red X and filename below
    base = kept.get("label") or (kept["path"].split("/")[1] if "/" in kept["path"] else "salmo")
    for i in range(n_removed):
        ax = fig.add_subplot(gs[0, 1 + i])
        img = Image.open(removed_paths[i]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        file_label = base + "-93.jpg" if i == 0 else base + "-93-{}.jpg".format(i + 1)
        ax.text(0.5, -0.08, file_label, transform=ax.transAxes, ha="center", fontsize=8, family="monospace")
        draw_red_x(ax, linewidth=3, color="red")

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", args.out)


if __name__ == "__main__":
    main()

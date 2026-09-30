"""Figures for the README, drawn from results/*.json.   python docs/make_figures.py  -> docs/img/*.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
GREY, TEAL, AMBER = "#8c959f", "#1a7f64", "#d4860b"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
load = lambda f: json.loads((ROOT / "results" / f).read_text())


def lp_scaling():
    R = load("final.json")
    lab = [f"{r['n'] / 1e6:.2f}M" if r["n"] >= 1e6 else f"{r['n'] / 1e3:.0f}k" for r in R]
    series = [("HiGHS (faster of simplex / IPM)", GREY, [r["highs"]["best"]["time"] for r in R]),
              ("SAMADHAN GPU, accurate (1e-6)", TEAL, [r["samadhan"]["1e-06"]["best"]["time"] for r in R]),
              ("SAMADHAN GPU, fast (1e-4)", AMBER, [r["samadhan"]["0.0001"]["best"]["time"] for r in R])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name, col, ys in series:
        ax.plot(lab, ys, marker="o", color=col, lw=2.2, label=name)
        ax.annotate(f"{ys[-1]:.0f} s" if ys[-1] >= 10 else f"{ys[-1]:.1f} s", (len(lab) - 1, ys[-1]),
                    textcoords="offset points", xytext=(8, -3), color=col, fontsize=10)
    ax.set_yscale("log"); ax.set_ylabel("solve time (s, log scale)"); ax.set_xlabel("refinery planning LP, variables")
    ax.set_title("LP: the lead grows with model size", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(IMG / "lp_scaling.png")


def miplib():
    R = [r for r in load("miplib.json") if "samadhan" in r]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lim = 60.0
    t = lambda d, ok: max(d["time"], 0.01) if ok else lim * 1.6
    for r in R:
        s_ok = r["samadhan"]["status"] == "optimal"
        h_ok = r["highs"]["status"] == "Optimal"
        x, y = t(r["highs"], h_ok), t(r["samadhan"], s_ok)
        col = TEAL if (s_ok and (not h_ok or y <= x)) else (AMBER if s_ok else GREY)
        ax.scatter(x, y, s=28, color=col, zorder=3, edgecolor="white", linewidth=0.5)
        if s_ok and (not h_ok or y < x):
            ax.annotate(r["name"], (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8, color=INK)
    ax.plot([0.01, lim * 2], [0.01, lim * 2], color=MUTED, lw=0.8, ls="--")
    ax.axhline(lim * 1.6, color=GRID, lw=0.8); ax.axvline(lim * 1.6, color=GRID, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.008, lim * 2.2); ax.set_ylim(0.008, lim * 2.2)
    ax.set_xlabel("HiGHS time (s)  —  right edge: not solved in 60 s")
    ax.set_ylabel("SAMADHAN C++ core time (s)  —  top edge: not solved")
    ax.set_title("MILP: MIPLIB 3, 60 s limit, 1 thread each\nbelow the diagonal = SAMADHAN faster",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(IMG / "miplib.png")


def qp():
    R = load("qp_maros.json")
    cats = {"both solved": 0, "only SAMADHAN": 0, "only HiGHS": 0, "neither": 0}
    for r in R:
        s = r.get("samadhan", {}).get("status") == "optimal"
        h = r.get("highs", {}).get("status") == "Optimal"
        cats["both solved" if s and h else "only SAMADHAN" if s else "only HiGHS" if h else "neither"] += 1
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    left = 0
    for (k, v), col in zip(cats.items(), [TEAL, AMBER, GREY, GRID]):
        ax.barh([0], [v], left=left, color=col, height=0.5)
        if v:
            ax.text(left + v / 2, 0, f"{k}\n{v}", ha="center", va="center", fontsize=9,
                    color="white" if col in (TEAL, GREY) else INK)
        left += v
    ax.set_xlim(0, left); ax.set_yticks([]); ax.grid(False); ax.set_xlabel(f"{left} Maros–Meszaros convex QPs")
    ax.set_title("QP: SAMADHAN (1e-6 relative KKT, CPU) vs HiGHS QP, 60 s limit", loc="left", color=INK,
                 fontsize=11, fontweight="bold")
    fig.savefig(IMG / "qp_maros.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    lp_scaling(); miplib(); qp()
    print("figures written to", IMG)

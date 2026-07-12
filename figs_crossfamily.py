# -*- coding: utf-8 -*-
"""figs_crossfamily.py — F18 跨家族稳健性图: DeepSeek-V3 vs Qwen-7B。
读 out/crossfamily.json, 分组条形展示四个核心指标在两个模型家族上的一致性。
运行: python3.13 figs_crossfamily.py
"""
import json, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"; plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": 0.8, "legend.frameon": False})

P = {"qwen": "#0F4D92", "ds": "#D9A441", "ink": "#222222", "n_mid": "#767676"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FIGS = os.path.join(OUT, "figs")


def main():
    cf = json.load(open(os.path.join(OUT, "crossfamily.json"), encoding="utf-8"))
    metrics = [("calib_r", "Calibration $r$"), ("abstain_gain", "Abstain recall gain"),
               ("frac_systematic", "Systematic\nomission share"), ("triage_auroc", "Triage AUROC")]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
    for ax, (key, lab) in zip(axes, metrics):
        rows = []
        for ds in ("zh", "en"):
            q = cf[ds].get("7b"); d = cf[ds].get("dsv3")
            rows.append((ds, q[key] if q else None, d[key] if d else None))
        x = np.arange(len(rows)); w = 0.36
        qv = [r[1] if r[1] is not None else 0 for r in rows]
        dv = [r[2] if r[2] is not None else 0 for r in rows]
        ax.bar(x-w/2, qv, w, color=P["qwen"], label="Qwen2.5-7B")
        ax.bar(x+w/2, dv, w, color=P["ds"], label="DeepSeek-V3")
        ax.set_xticks(x); ax.set_xticklabels(["ZH", "EN"], fontsize=7)
        ax.set_title(lab, fontsize=7)
        ax.axhline(0, color=P["n_mid"], lw=0.6)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Cross-family robustness: SCOPE conclusions reproduce on an unrelated model architecture",
                 fontsize=7.8, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "F18_crossfamily.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "F18_crossfamily.png"), dpi=200, bbox_inches="tight")
    print("OK F18_crossfamily")


if __name__ == "__main__":
    main()

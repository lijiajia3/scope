# -*- coding: utf-8 -*-
"""fig_mechanism.py — 新增图: 为何 entropy/consistency 基线在崩溃句上反向。
崩溃句(漏抽最重)上 SCOPE 覆盖风险高(正确预警), 而两基线读作低风险(错误放行)。
读 out/mechanism_auroc.json。输出 Fx_mechanism.pdf/png。
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

P = {"blue": "#0F4D92", "violet": "#9A4D8E", "gold": "#D9A441", "red": "#B64342",
     "mut": "#6B7280", "ink": "#222222"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FIGS = os.path.join(OUT, "figs")

d = json.load(open(os.path.join(OUT, "mechanism_auroc.json"), encoding="utf-8"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, ds, lab in zip(axes, ("en", "zh"), ("CoNLL-EN", "Clinical-ZH")):
    col = d[ds]["collapse"]; hea = d[ds]["healthy"]
    sigs = [("SCOPE\ncoverage risk", "cov_risk", P["blue"]),
            ("semantic\nentropy", "sem_entropy", P["violet"]),
            ("1 - self-\nconsistency", None, P["gold"])]
    x = np.arange(3); w = 0.38
    col_vals = [col["cov_risk"], col["sem_entropy"], round(1-col["self_cons"], 4)]
    hea_vals = [hea["cov_risk"], hea["sem_entropy"], round(1-hea["self_cons"], 4)]
    ax.bar(x - w/2, col_vals, w, color=P["red"], alpha=0.85,
           label=f"collapse sentences (miss rate {col['miss_rate']:.2f})")
    ax.bar(x + w/2, hea_vals, w, color=P["mut"], alpha=0.7,
           label=f"healthy sentences (miss rate {hea['miss_rate']:.2f})")
    for xi, (cv, hv) in enumerate(zip(col_vals, hea_vals)):
        ax.text(xi - w/2, cv + 0.02, f"{cv:.2f}", ha="center", fontsize=5.6, color=P["red"])
        ax.text(xi + w/2, hv + 0.02, f"{hv:.2f}", ha="center", fontsize=5.6, color=P["mut"])
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sigs], fontsize=6.4)
    ax.set_ylabel("Estimated risk (higher = flag)"); ax.set_ylim(0, 1.05)
    ax.set_title(lab, fontsize=7.6)
    ax.legend(fontsize=5.5, loc="upper center")
    # 标注反向
    ax.annotate("baselines read\ncollapse as safe", xy=(1, col_vals[1]+0.02),
                xytext=(1.3, 0.55), fontsize=5.6, color=P["violet"],
                arrowprops=dict(arrowstyle="->", color=P["violet"], lw=0.8))
axes[0].text(-0.14, 1.06, "a", transform=axes[0].transAxes, fontsize=10, fontweight="bold")
axes[1].text(-0.14, 1.06, "b", transform=axes[1].transAxes, fontsize=10, fontweight="bold")
fig.suptitle("Content-reading baselines rank the most incomplete sentences as the most reliable",
             fontsize=7.8, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "Fx_mechanism.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "Fx_mechanism.png"), dpi=200, bbox_inches="tight")
print("OK Fx_mechanism")

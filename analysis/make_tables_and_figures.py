#!/usr/bin/env python3
"""Regenerate every table and figure of the study from the saved run results.

Inputs (all in this repository):
  results/run1_ablation/, results/run2_baselines_losses/, results/run3_noise_efficiency/   (results.jsonl + preds/*.npy)
  data/splits.csv                                                                              (image_name, split, group)

The test labels are NOT stored in the repository (the images and their labels belong to the Cassava dataset). The script
reads them from your local copy of the data: pass --data-dir pointing either to
  (a) a folder with one sub-folder per class (e.g. Cassava___mosaic_disease/*.jpg), or
  (b) the competition folder that contains train.csv (image_id,label).
Only file names and labels are read; no image is opened.

Outputs: results/summary/*.json|csv  and  figures/*.png
Run:  python analysis/make_tables_and_figures.py --data-dir /path/to/cassava
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUN_DIRS = [RES / "run1_ablation", RES / "run2_baselines_losses", RES / "run3_noise_efficiency"]
NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
KEYWORDS = [("healthy", 4), ("bacterial", 0), ("cbb", 0), ("streak", 1), ("brown", 1), ("cbsd", 1),
            ("mottle", 2), ("green", 2), ("cgm", 2), ("mosaic", 3), ("cmd", 3)]


def class_of(folder):
    n = folder.lower()
    for k, v in KEYWORDS:
        if k in n:
            return v
    return None


def load_labels(data_dir):
    """file name -> label, from train.csv (image_id,label) or from class sub-folders."""
    data_dir = Path(data_dir)
    for csv in data_dir.rglob("train.csv"):
        df = pd.read_csv(csv)
        if {"image_id", "label"} <= set(df.columns):
            return dict(zip(df.image_id, df.label.astype(int)))
    labels = {}
    for dp, _, fn in os.walk(data_dir):
        cls = None
        for anc in [Path(dp)] + list(Path(dp).parents):
            cls = class_of(anc.name)
            if cls is not None or anc == data_dir:
                break
        if cls is None:
            continue
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                labels[f] = cls
    if not labels:
        raise SystemExit("No labels found under --data-dir (expected train.csv or class sub-folders).")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    (ROOT / "figures").mkdir(exist_ok=True)
    (RES / "summary").mkdir(exist_ok=True)

    lab = load_labels(args.data_dir)
    sp = pd.read_csv(ROOT / "data" / "splits.csv")
    missing = [n for n in sp.image_name if n not in lab]
    if missing:
        raise SystemExit(f"{len(missing)} images of the split file were not found in --data-dir (e.g. {missing[:3]}).")
    sp["label"] = sp.image_name.map(lab).astype(int)
    y = sp[sp.split == "test"].label.to_numpy()
    print("test images:", len(y), "class counts:", np.bincount(y, minlength=5).tolist())

    R = {}
    for d in RUN_DIRS:
        for line in open(d / "results.jsonl"):
            r = json.loads(line); R[r["run_id"]] = r

    def P(rid):
        for d in RUN_DIRS:
            f = d / "preds" / f"{rid}.npy"
            if f.exists():
                return np.load(f).astype(int)
        raise FileNotFoundError(rid)

    # sanity: stored predictions must reproduce the recorded test metrics
    for rid in ["abl_base__s0", "bl_convnext_tiny__s0", "noise30_sce__s1"]:
        assert abs(f1_score(y, P(rid), average="macro") - R[rid]["test"]["macro_f1"]) < 1e-9, f"labels do not match results for {rid}"
    print("label check passed: stored predictions reproduce the recorded macro-F1")

    LABEL = {"abl_base": "Base (GAP head)", "abl_A": "A: LANet head", "abl_L": "L: logit-adj.", "abl_N": "N: noise module",
             "abl_AL": "A+L", "abl_AN": "A+N", "abl_LN": "L+N", "abl_full": "A+L+N (full)",
             "bl_mobilenetv3_large_100": "MobileNetV3-L (std head)", "bl_ghostnet_100": "GhostNet-100", "bl_efficientnet_b0": "EfficientNet-B0",
             "bl_mobilevit_s": "MobileViT-S", "bl_convnext_tiny": "ConvNeXt-T", "loss_focal": "Base + focal", "loss_cb": "Base + class-balanced"}
    runs = lambda prefix: [R[f"{prefix}__s{s}"] for s in range(3)]

    # ---------------- summary table (Runs 1 and 2) ----------------
    rows = []
    for k in LABEL:
        rs = runs(k)
        t = lambda g: np.array([g(r["test"]) for r in rs]) * 100
        rows.append(dict(key=k, name=LABEL[k], acc_m=t(lambda x: x["acc"]).mean(), acc_s=t(lambda x: x["acc"]).std(ddof=1),
                         f1_m=t(lambda x: x["macro_f1"]).mean(), f1_s=t(lambda x: x["macro_f1"]).std(ddof=1),
                         bal_m=t(lambda x: x["bal_acc"]).mean(), bal_s=t(lambda x: x["bal_acc"]).std(ddof=1),
                         cbb_m=t(lambda x: x["recall"][0]).mean(), cbb_s=t(lambda x: x["recall"][0]).std(ddof=1),
                         params=rs[0]["params_m"], gmacs=rs[0]["gmacs"], minutes=np.mean([r["train_s"] for r in rs]) / 60,
                         best_ep=np.mean([r["best_epoch"] for r in rs]), ep_run=np.mean([r["epochs_run"] for r in rs]),
                         val_f1=np.mean([r["val_macro_f1"] for r in rs]) * 100, seeds_f1=[r["test"]["macro_f1"] * 100 for r in rs]))
    T = pd.DataFrame(rows)
    T.to_json(RES / "summary" / "summary.json", orient="records", indent=1)
    T.drop(columns="seeds_f1").round(3).to_csv(RES / "summary" / "table_runs1_2_summary.csv", index=False)

    # ---------------- paired bootstrap + McNemar vs. base ----------------
    rng = np.random.RandomState(0); B = 2000
    idx = [rng.randint(0, len(y), len(y)) for _ in range(B)]
    mf = lambda ps, ix: np.mean([f1_score(y[ix], p[ix], average="macro") for p in ps])
    pb = [P(f"abl_base__s{s}") for s in range(3)]
    mb = np.array([mf(pb, ix) for ix in idx])
    stats = []
    for k in LABEL:
        if k == "abl_base":
            continue
        ps = [P(f"{k}__s{s}") for s in range(3)]
        d = np.array([mf(ps, ix) for ix in idx]) - mb
        dm = (np.mean([f1_score(y, p, average="macro") for p in ps]) - np.mean([f1_score(y, p, average="macro") for p in pb])) * 100
        mc = []
        for s in range(3):
            a, b = pb[s], ps[s]; n01 = int(((a == y) & (b != y)).sum()); n10 = int(((a != y) & (b == y)).sum())
            mc.append(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)
        stats.append(dict(key=k, name=LABEL[k], d=dm, lo=np.percentile(d, 2.5) * 100, hi=np.percentile(d, 97.5) * 100, p=mc))
    S = pd.DataFrame(stats); S.to_json(RES / "summary" / "stats.json", orient="records", indent=1)

    pc = {}
    for k in LABEL:
        ps = [P(f"{k}__s{s}") for s in range(3)]
        pc[k] = dict(f1=(np.mean([f1_score(y, p, average=None, labels=range(5)) for p in ps], 0) * 100).tolist(),
                     rec=(np.mean([recall_score(y, p, average=None, labels=range(5)) for p in ps], 0) * 100).tolist(),
                     prec=(np.mean([precision_score(y, p, average=None, labels=range(5)) for p in ps], 0) * 100).tolist())
    json.dump(pc, open(RES / "summary" / "perclass.json", "w"), indent=1)

    # ---------------- Run 3: label noise ----------------
    rates = [10, 20, 30]; methods = ["ce", "AL", "full", "sce"]
    out = {"rows": {}, "stats": {}, "weights": {}, "clean": {}}
    for m in methods:
        for r in rates:
            rs = [R[f"noise{r}_{m}__s{s}"] for s in range(2)]
            g = lambda f: (np.array([f(x["test"]) for x in rs]) * 100).tolist()
            out["rows"][f"{m}@{r}"] = dict(f1=g(lambda t: t["macro_f1"]), acc=g(lambda t: t["acc"]), bal=g(lambda t: t["bal_acc"]),
                                           cbb=g(lambda t: t["recall"][0]), best_ep=[x["best_epoch"] for x in rs], ep_run=[x["epochs_run"] for x in rs],
                                           minutes=[x["train_s"] / 60 for x in rs], val=[x["val_macro_f1"] * 100 for x in rs])
    for m, k in [("ce", "abl_base"), ("AL", "abl_AL"), ("full", "abl_full")]:
        rs = [R[f"{k}__s{s}"] for s in range(2)]
        out["clean"][m] = dict(f1=[x["test"]["macro_f1"] * 100 for x in rs], bal=[x["test"]["bal_acc"] * 100 for x in rs], cbb=[x["test"]["recall"][0] * 100 for x in rs])
    for r in rates:
        rs = [R[f"noise{r}_full__s{s}"] for s in range(2)]
        out["weights"][str(r)] = dict(clean=[x["nz_w_clean"] for x in rs], noisy=[x["nz_w_noisy"] for x in rs], actual=[x["noise_actual"] for x in rs])

    def paired(pa, pm):
        d = np.array([mf(pm, ix) for ix in idx]) - np.array([mf(pa, ix) for ix in idx])
        dm = (np.mean([f1_score(y, p, average="macro") for p in pm]) - np.mean([f1_score(y, p, average="macro") for p in pa])) * 100
        mc = []
        for s in range(2):
            a, b = pa[s], pm[s]; n01 = int(((a == y) & (b != y)).sum()); n10 = int(((a != y) & (b == y)).sum())
            mc.append(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)
        return dict(d=dm, lo=float(np.percentile(d, 2.5) * 100), hi=float(np.percentile(d, 97.5) * 100), p=mc)

    for r in rates:
        pc_ = [P(f"noise{r}_ce__s{s}") for s in range(2)]
        for m in ["AL", "full", "sce"]:
            out["stats"][f"{m}@{r}"] = paired(pc_, [P(f"noise{r}_{m}__s{s}") for s in range(2)])
        out["stats"][f"full-vs-AL@{r}"] = paired([P(f"noise{r}_AL__s{s}") for s in range(2)], [P(f"noise{r}_full__s{s}") for s in range(2)])
    json.dump(out, open(RES / "summary" / "noise.json", "w"), indent=1)
    eff = pd.read_csv(RES / "run3_noise_efficiency" / "table_efficiency.csv")
    eff.to_json(RES / "summary" / "eff.json", orient="records", indent=1)

    # ---------------- figures ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, RED, GREY, GREEN, ORANGE = "#2F5D9B", "#B03A3A", "#8A8A8A", "#4C8C3A", "#D98A1F"
    Fig = ROOT / "figures"

    order = ["abl_A", "abl_L", "abl_N", "abl_AL", "abl_AN", "abl_LN", "abl_full", "loss_focal", "loss_cb", "bl_mobilenetv3_large_100",
             "bl_ghostnet_100", "bl_efficientnet_b0", "bl_mobilevit_s", "bl_convnext_tiny"]
    Sx = S.set_index("key").loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    ypos = np.arange(len(order))[::-1]
    for yy, (k, r) in zip(ypos, Sx.iterrows()):
        sig = (r.lo > 0) or (r.hi < 0); col = (GREEN if r.d > 0 else RED) if sig else GREY
        ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=2); ax.plot(r.d, yy, "o", color=col, ms=5)
    ax.axvline(0, color="black", lw=0.8); ax.set_yticks(ypos); ax.set_yticklabels([LABEL[k] for k in order])
    ax.axhline(6.5, color="#cccccc", lw=0.6, ls="--"); ax.axhline(4.5, color="#cccccc", lw=0.6, ls="--")
    ax.set_xlabel("Change in test macro-F1 vs. the base model (percentage points)\nseed-averaged; bars = 95% bootstrap CI over test images", fontsize=8.5)
    plt.tight_layout(); plt.savefig(Fig / "forest.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=200); Tk = T.set_index("key")
    for k, r in Tk.iterrows():
        if k.startswith("loss"):
            continue
        ours = k.startswith("abl")
        ax.errorbar(r.gmacs, r.f1_m, yerr=r.f1_s, fmt="o" if ours else "s", color=BLUE if ours else ORANGE, ms=6, alpha=0.85, capsize=2, lw=1)
    def lab_(txt, xy, xyt):
        ax.annotate(txt, xy, xyt, fontsize=7.5, ha="left", va="center", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
    lab_("Ablation variants (same backbone):\nbase 73.1, full 72.5, L 71.0", (0.22, 73.2), (0.30, 75.2))
    lab_("GhostNet-100", (Tk.loc["bl_ghostnet_100"].gmacs, 71.4), (0.115, 69.4))
    lab_("MobileNetV3-L (standard head)", (0.22, 68.6), (0.27, 67.2))
    lab_("EfficientNet-B0", (0.38, 72.7), (0.45, 70.9)); lab_("MobileViT-S", (1.52, 74.4), (1.05, 72.6)); lab_("ConvNeXt-T", (4.45, 76.0), (2.7, 77.2))
    ax.set_xscale("log"); ax.set_xlim(0.1, 6.5); ax.set_ylim(66, 78.5)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5])); ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.5", "1", "2", "5"])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Computation (GMACs at 224×224, log scale)"); ax.set_ylabel("Test macro-F1 (%), mean ± SD over 3 seeds")
    ax.plot([], [], "o", color=BLUE, label="Ablation variants (same backbone)"); ax.plot([], [], "s", color=ORANGE, label="Standard timm baselines")
    ax.legend(frameon=False, loc="lower right", fontsize=8); plt.tight_layout(); plt.savefig(Fig / "pareto.png"); plt.close()

    sel = ["abl_base", "abl_L", "abl_full", "bl_convnext_tiny"]; cols = [GREY, RED, BLUE, GREEN]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=200, sharey=True)
    for ax, (m, ttl) in zip(axes, [("rec", "Recall"), ("prec", "Precision"), ("f1", "F1")]):
        w = 0.2
        for i, (k, c) in enumerate(zip(sel, cols)):
            ax.bar(np.arange(5) + (i - 1.5) * w, pc[k][m], w, color=c, label=LABEL[k])
        ax.set_xticks(range(5)); ax.set_xticklabels(NAMES, fontsize=8); ax.set_title(ttl, fontsize=9); ax.set_ylim(40, 100)
    axes[0].set_ylabel("%"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(Fig / "perclass.png"); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=200)
    for ax, k in zip(axes, ["abl_base", "abl_L", "bl_convnext_tiny"]):
        cm = sum(confusion_matrix(y, P(f"{k}__s{s}"), labels=range(5)) for s in range(3)).astype(float); cmn = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 55 else "black")
        ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(NAMES, fontsize=7, rotation=30); ax.set_yticklabels(NAMES, fontsize=7)
        ax.set_title(LABEL[k], fontsize=8); ax.set_xlabel("Predicted", fontsize=8)
        if k == "abl_base":
            ax.set_ylabel("True (row-normalised %)", fontsize=8)
    plt.tight_layout(); plt.savefig(Fig / "confusion.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    ab = ["abl_base", "abl_A", "abl_L", "abl_N", "abl_AL", "abl_AN", "abl_LN", "abl_full"]
    be = [np.mean([r["best_epoch"] for r in runs(k)]) for k in ab]; er = [np.mean([r["epochs_run"] for r in runs(k)]) for k in ab]
    x = np.arange(len(ab)); ax.bar(x - 0.2, be, 0.4, color=BLUE, label="Best epoch (val macro-F1)"); ax.bar(x + 0.2, er, 0.4, color="#BBBBBB", label="Epochs run (early stop)")
    ax.set_xticks(x); ax.set_xticklabels(["base", "A", "L", "N", "AL", "AN", "LN", "full"]); ax.set_ylabel("Epoch (mean of 3 seeds)"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(Fig / "epochs.png"); plt.close()

    col = {"ce": GREY, "AL": ORANGE, "full": BLUE, "sce": GREEN}
    lab = {"ce": "Base + CE", "AL": "LANet + logit-adj. (A+L)", "full": "Full (A+L+N)", "sce": "Base + symmetric CE"}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), dpi=200)
    for ax, (key, ttl) in zip(axes, [("f1", "Macro-F1"), ("bal", "Balanced accuracy"), ("cbb", "CBB recall")]):
        for m in methods:
            xs = [0] if m in out["clean"] else []; ys = [np.mean(out["clean"][m][key])] if m in out["clean"] else []
            xs += rates; ys += [np.mean(out["rows"][f"{m}@{r}"][key]) for r in rates]
            ax.plot(xs, ys, "-o", color=col[m], label=lab[m], ms=4, lw=1.5)
            for r in rates:
                v = out["rows"][f"{m}@{r}"][key]; ax.plot([r, r], [min(v), max(v)], color=col[m], lw=0.8, alpha=0.6)
        ax.set_xticks([0] + rates); ax.set_xticklabels(["0\n(clean)", "10", "20", "30"]); ax.set_xlabel("Injected label noise (%)"); ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel("Test (%), mean of 2 seeds; vertical lines = range")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(Fig / "noise.png"); plt.close()

    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200); x = np.arange(3); w = 0.35
    ax.bar(x - w / 2, [np.mean(out["weights"][str(r)]["clean"]) for r in rates], w, color=BLUE, label="Clean labels")
    ax.bar(x + w / 2, [np.mean(out["weights"][str(r)]["noisy"]) for r in rates], w, color=RED, label="Flipped labels")
    ax.set_xticks(x); ax.set_xticklabels(["10%", "20%", "30%"]); ax.set_xlabel("Injected label noise"); ax.set_ylabel("Mean sample weight"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); plt.tight_layout(); plt.savefig(Fig / "weights.png"); plt.close()

    print("done: tables in results/summary, figures in figures/")


if __name__ == "__main__":
    main()

# =====================================================================================
#  SHARED CORE  (this block is identical in Run 1, Run 2 and Run 3 -- do not edit one copy only)
#  Cassava Leaf Disease | lightweight, explainable, imbalance- and noise-aware classifier
#  Target: Kaggle T4 (16 GB) | 4 CPU cores | one 12-hour session per run
# =====================================================================================
import os, sys, io, json, time, math, random, copy, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, recall_score, precision_score

warnings.filterwarnings("ignore")
T0 = time.time()

# ----------------------------------- switches ----------------------------------------
SMOKE = False                      # True = ~5-minute dry run on a 1,500-image subset (run this first!)
USE_PRETRAINED = os.environ.get("NO_PRETRAINED", "0") != "1"

# ----------------------------------- paths -------------------------------------------
IS_KAGGLE = Path("/kaggle").exists()
INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path("./input")
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("./outputs")
TMP_DIR = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
for _d in (OUT_DIR, OUT_DIR / "preds", OUT_DIR / "ckpt"):
    _d.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_DIR / ("results_smoke.jsonl" if SMOKE else "results.jsonl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True

# ----------------------------------- hyper-parameters --------------------------------
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
N_CLS = 5
HP = dict(
    cache=256, img=224, bs=64, epochs=35, patience=8, min_epochs=12, warmup_steps=200,
    wd=0.02, drop=0.2, lam_att=0.05,                      # attention sparsity weight
    nz_q=0.8, nz_warm=3, nz_temp=1.0, nz_beta=1.0,        # noise module (quantile, warm-up epochs, temperature, attention coupling)
    time_budget_h=10.5,                                   # stop starting new runs after this (hard limit is 12 h)
    split_seed=42,
)
if SMOKE:
    HP.update(epochs=2, min_epochs=1, patience=2, warmup_steps=5, time_budget_h=1.0)
PROPOSED_BACKBONE = "mobilenetv3_large_100"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# ablation configs: name -> (use_attention_block, loss_type, use_noise_module)
ABL_CONFIGS = {
    "base": (False, "ce", False), "A": (True, "ce", False), "L": (False, "la", False), "N": (False, "ce", True),
    "AL": (True, "la", False), "AN": (True, "ce", True), "LN": (False, "la", True), "full": (True, "la", True),
}


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ================================ file / result bookkeeping ============================
def walk_inputs(max_depth=3):
    if not INPUT_ROOT.exists():
        return
    for dp, dn, fn in os.walk(INPUT_ROOT):
        depth = len(Path(dp).relative_to(INPUT_ROOT).parts)
        if depth >= max_depth:
            dn[:] = []
        yield Path(dp), fn


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def class_from_name(name):
    """Folder name -> class index (0 CBB, 1 CBSD, 2 CGM, 3 CMD, 4 Healthy) by keyword; None if no match."""
    n = name.lower()
    if "healthy" in n: return 4
    if "bacterial" in n or "cbb" in n: return 0
    if "streak" in n or "brown" in n or "cbsd" in n: return 1
    if "mottle" in n or "green" in n or "cgm" in n: return 2
    if "mosaic" in n or "cmd" in n: return 3
    return None


def load_cassava_table():
    """Returns DataFrame(image_id, label, path). Supports (1) the original competition layout (train.csv + train_images/)
       and (2) class-folder datasets (one sub-folder per disease, e.g. nirmalsankalana/cassava-leaf-disease-dataset)."""
    # ---- (1) CSV layout ----
    for dp, fn in walk_inputs(4):
        for f in sorted(fn, key=lambda n: (n != "train.csv", n)):
            if not f.lower().endswith(".csv"):
                continue
            try:
                df = pd.read_csv(dp / f)
            except Exception:
                continue
            if {"image_id", "label"} <= set(df.columns) and len(df) > 100:
                first = str(df["image_id"].iloc[0])
                for sub in ("train_images", "train", "images", ""):
                    if (dp / sub / first).exists():
                        df["path"] = [str(dp / sub / i) for i in df["image_id"]]
                        log("layout: competition CSV ->", dp / f); return df[["image_id", "label", "path"]]
    # ---- (2) class-folder layout ----
    by_dir = {}
    for dp, dn, fn in os.walk(INPUT_ROOT):
        imgs = [f for f in fn if f.lower().endswith(IMG_EXT)]
        if imgs:
            by_dir[Path(dp)] = imgs
    if by_dir:
        rows, unmatched = [], {}
        for d, imgs in by_dir.items():
            cls = None
            for anc in [d] + list(d.parents):                      # folder name first, then parents (e.g. .../CMD/images/)
                if anc == INPUT_ROOT: break
                cls = class_from_name(anc.name)
                if cls is not None: break
            if cls is None:
                unmatched[d] = len(imgs); continue
            rows += [(str(d.relative_to(INPUT_ROOT) / f), cls, str(d / f)) for f in imgs]
        if rows and len({r[1] for r in rows}) == N_CLS:
            df = pd.DataFrame(rows, columns=["image_id", "label", "path"])
            log(f"layout: class folders ({len(df)} images). Per-class counts {np.bincount(df.label, minlength=N_CLS).tolist()} "
                f"(order {CLASS_NAMES}). Folders -> class mapping:")
            for d in sorted({Path(p).parent for p in df.path}):
                log(f"    {d.relative_to(INPUT_ROOT)}  ->  {CLASS_NAMES[class_from_name(d.name) if class_from_name(d.name) is not None else df[df.path.str.startswith(str(d))].label.iloc[0]]}  ({len(by_dir[d])} images)")
            if unmatched:
                log("    ignored image folders with no matching class name:", {str(k): v for k, v in list(unmatched.items())[:8]})
            return df
        # fallback: exactly 5 leaf folders with unrecognised names -> alphabetical (VERIFY the mapping printed below!)
        leaves = {d.name: d for d in by_dir}
        if len(leaves) == N_CLS and sum(len(v) for v in by_dir.values()) > 1000:
            log("!! WARNING: folder names not recognised; mapping alphabetically -> verify against", CLASS_NAMES)
            rows = []
            for k, nm in enumerate(sorted(leaves)):
                log(f"    {nm} -> class {k} ({CLASS_NAMES[k]})  ({len(by_dir[leaves[nm]])} images)")
                rows += [(str(leaves[nm].relative_to(INPUT_ROOT) / f), k, str(leaves[nm] / f)) for f in by_dir[leaves[nm]]]
            return pd.DataFrame(rows, columns=["image_id", "label", "path"])
    log("Cassava data NOT found. Contents of /kaggle/input (3 levels):")
    for dp, fn in walk_inputs(3):
        log("   ", dp, "| files:", fn[:6], "..." if len(fn) > 6 else "")
    raise FileNotFoundError("Cassava data not found. Attach either the competition 'Cassava Leaf Disease Classification' "
                            "(train.csv + train_images/) or a dataset with one sub-folder per disease class.")


def load_records():
    """All finished runs: this session + any previous-run outputs attached as inputs."""
    recs = {}
    paths = [] if SMOKE else [dp / "results.jsonl" for dp, fn in walk_inputs(4) if "results.jsonl" in fn]
    if RESULTS.exists():
        paths.append(RESULTS)
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line); recs[r["run_id"]] = r
    return recs


def append_result(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def find_artifact(sub, name):
    """Locate OUT_DIR/sub/name or <any attached input>/sub/name (e.g. ckpt/abl_full__s0.pt)."""
    p = OUT_DIR / sub / name
    if p.exists():
        return p
    for dp, fn in walk_inputs(5):
        if dp.name == sub and name in fn:
            return dp / name
    return None


class Budget:
    def __init__(self, hours):
        self.limit = hours * 3600; self.hist = {}

    def elapsed(self):
        return time.time() - T0

    def add(self, kind, secs):
        self.hist.setdefault(kind, []).append(secs)

    def estimate(self, kind, default_s):
        h = self.hist.get(kind)
        return max(h) * 1.15 if h else default_s

    def allows(self, kind, default_s):
        return self.elapsed() + self.estimate(kind, default_s) <= self.limit


BUDGET = Budget(HP["time_budget_h"])


# ================================ data: cache, de-duplicated split =====================
def _load_resize(path, size):
    im = Image.open(path)
    im.draft("RGB", (size, size))                        # fast JPEG DCT-domain down-scaling
    return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))


def build_cache(paths, size):
    key = __import__("hashlib").md5("|".join(paths).encode()).hexdigest()[:8]
    cache = TMP_DIR / f"cassava_{size}_{len(paths)}_{key}.npy"
    if cache.exists():
        log("loading cached array", cache); return np.load(cache)
    log(f"decoding + resizing {len(paths)} images to {size}x{size} ...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        X = np.stack(list(ex.map(lambda p: _load_resize(Path(p), size), paths)))
    np.save(cache, X)
    return X


@torch.no_grad()
def near_duplicate_groups(X, max_ham=2, chunk=2048):
    """64-bit dHash + Hamming <= max_ham -> connected components = groups (kept together in one split)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = len(X); bits = []
    for i in range(0, N, chunk):
        xb = torch.from_numpy(X[i:i + chunk]).to(DEVICE).permute(0, 3, 1, 2).float()
        g = (0.299 * xb[:, 0:1] + 0.587 * xb[:, 1:2] + 0.114 * xb[:, 2:3])
        g = F.adaptive_avg_pool2d(g, (8, 9))
        bits.append((g[..., 1:] > g[..., :-1]).flatten(1))
    B = torch.cat(bits).float() * 2 - 1                      # +-1, shape N x 64
    B = B.half() if USE_CUDA else B
    rows, cols = [], []
    for i in range(0, N, chunk):
        ham = (64 - (B[i:i + chunk] @ B.T).float()) / 2
        r, c = torch.nonzero(ham <= max_ham, as_tuple=True)
        r = r + i; keep = r < c
        rows.append(r[keep].cpu().numpy()); cols.append(c[keep].cpu().numpy())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    _, lab = connected_components(adj, directed=False)
    return lab


class Data:  # simple container
    pass


def prepare_data():
    df = load_cassava_table()
    if SMOKE:
        df = df.sample(min(1500, len(df)), random_state=0).reset_index(drop=True)
    ids, y = df["image_id"].tolist(), df["label"].to_numpy().astype(int)
    log(f"dataset: {len(df)} images, class counts {np.bincount(y, minlength=N_CLS).tolist()}")
    X = build_cache(df["path"].tolist(), HP["cache"])
    groups = near_duplicate_groups(X)
    n_dup = int((pd.Series(groups).map(pd.Series(groups).value_counts()) > 1).sum())
    log(f"near-duplicate check: {n_dup} images belong to duplicate groups (kept inside one split)")
    folds = [f for _, f in StratifiedGroupKFold(10, shuffle=True, random_state=HP["split_seed"]).split(X[:, 0, 0, 0], y, groups)]
    te, va = folds[0], folds[1]
    tr = np.setdiff1d(np.arange(len(y)), np.concatenate([te, va]))
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[tr]) & set(groups[va]))
    split = np.array(["train"] * len(y), dtype=object); split[va] = "val"; split[te] = "test"
    pd.DataFrame(dict(image_id=ids, label=y, split=split, group=groups)).to_csv(OUT_DIR / "splits.csv", index=False)
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        log(f"  {nm:5s} n={len(ix):6d} class counts={np.bincount(y[ix], minlength=N_CLS).tolist()}")
    d = Data()
    d.split_hash = __import__("hashlib").md5(te.tobytes() + y[te].tobytes()).hexdigest()[:10]
    put = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)           # uint8 N,H,W,3 stays on the GPU
    d.Xtr, d.Xva, d.Xte = put(X[tr]), put(X[va]), put(X[te])
    d.ytr, d.yva, d.yte = [torch.from_numpy(y[i]).long().to(DEVICE) for i in (tr, va, te)]
    d.yte_np, d.yva_np = y[te], y[va]
    d.Xte_np = X[te].copy()                                                       # CPU copy (JPEG corruption, saliency)
    d.n_train = len(tr)
    del X
    return d


# ================================ GPU augmentation / pre-processing ====================
def to_float(xb_u8):                      # uint8 N,H,W,3 -> float N,3,H,W in [0,1]
    return xb_u8.permute(0, 3, 1, 2).float().div_(255.0)


def norm(x):
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def prep_eval(x01):                       # full image, resized 256 -> 224
    return norm(F.interpolate(x01, size=HP["img"], mode="bilinear", antialias=True, align_corners=False))


def train_aug(x, out=None):
    """random-resized-crop + flip + +-15 deg rotation (one affine grid_sample) + brightness/contrast/saturation."""
    out = out or HP["img"]; B, dev = x.size(0), x.device
    area = torch.empty(B, device=dev).uniform_(0.35, 1.0)
    ar = torch.exp(torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)))
    sx, sy = torch.sqrt(area * ar).clamp(max=1.0), torch.sqrt(area / ar).clamp(max=1.0)
    ang = torch.empty(B, device=dev).uniform_(-math.pi / 12, math.pi / 12)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    tx, ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - sx), (torch.rand(B, device=dev) * 2 - 1) * (1 - sy)
    c, s = torch.cos(ang), torch.sin(ang)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0], th[:, 0, 1], th[:, 0, 2] = sx * c * flip, -sy * s, tx
    th[:, 1, 0], th[:, 1, 1], th[:, 1, 2] = sx * s * flip, sy * c, ty
    grid = F.affine_grid(th, (B, 3, out, out), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    u = lambda: torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    x = x * u()
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * u() + m
    g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]; x = (x - g) * u() + g     # no hue shift: colour = symptom
    return x.clamp_(0, 1)


# ================================ models ================================================
def find_weights(name):
    for dp, fn in walk_inputs(4):
        for f in fn:
            if name in f and f.endswith((".pth", ".pt", ".safetensors", ".bin")):
                return dp / f
    return None


def create_timm(name, pretrained=None, **kw):
    pretrained = USE_PRETRAINED if pretrained is None else pretrained
    if not pretrained:
        return timm.create_model(name, pretrained=False, **kw)
    try:
        return timm.create_model(name, pretrained=True, **kw)
    except Exception as e:
        w = find_weights(name)
        if w is None:
            raise RuntimeError(f"Could not download pretrained weights for '{name}' ({e}). Either switch Internet ON in the "
                               f"notebook settings, or attach a dataset containing the weights file.")
        log(f"using local weights {w}")
        return timm.create_model(name, pretrained=True, pretrained_cfg_overlay=dict(file=str(w)), **kw)


class CBA(nn.Sequential):
    def __init__(self, i, o, k=1, groups=1):
        super().__init__(nn.Conv2d(i, o, k, padding=k // 2, groups=groups, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))


class BaselineNet(nn.Module):
    """Same light backbone as the proposed model, plain global-average-pooling head."""
    def __init__(self, backbone=PROPOSED_BACKBONE, pretrained=None, drop=0.2):
        super().__init__()
        self.enc = create_timm(backbone, pretrained, features_only=True, out_indices=(3, 4))
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.feature_info.channels()[-1], N_CLS))

    def encode(self, x):
        return self.enc(x)

    def head(self, feats):
        f = feats[-1]
        return self.fc(f.mean((2, 3))), {"cam": f}

    def forward(self, x, extra=False):
        l, e = self.head(self.enc(x)); return (l, e) if extra else l

    def head_parameters(self):
        return list(self.fc.parameters())


class LANet(nn.Module):
    """Lesion-Attention Network: local-global fusion + a spatial attention map that is part of the forward pass.
       stride-16 features (local detail) are gated by projected stride-32 features (global context); a 1x1 attention head
       produces a lesion map A in [0,1] (14x14); the classifier sees attention-pooled + average-pooled fused features."""
    def __init__(self, backbone=PROPOSED_BACKBONE, pretrained=None, d=128, drop=0.2):
        super().__init__()
        self.enc = create_timm(backbone, pretrained, features_only=True, out_indices=(3, 4))
        c1, c2 = self.enc.feature_info.channels()
        self.p1, self.p2 = CBA(c1, d, 1), CBA(c2, d, 1)
        self.dw = CBA(d, d, 3, groups=d)
        self.gate = nn.Sequential(nn.Conv2d(d, d, 1), nn.Sigmoid())
        self.att = nn.Sequential(nn.Conv2d(d, d // 4, 1), nn.SiLU(inplace=True), nn.Conv2d(d // 4, 1, 1))
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * d, N_CLS))

    def encode(self, x):
        return self.enc(x)

    def head(self, feats):
        f1, f2 = feats
        loc = self.dw(self.p1(f1))
        glo = F.interpolate(self.p2(f2), size=loc.shape[-2:], mode="bilinear", align_corners=False)
        fused = loc * self.gate(glo) + glo
        a = torch.sigmoid(self.att(fused))                                   # B,1,H,W  lesion map
        w = a / (a.sum((2, 3), keepdim=True) + 1e-6)                         # normalised weights (sum to 1)
        pooled = (fused * w).sum((2, 3))
        logits = self.fc(torch.cat([pooled, fused.mean((2, 3))], 1))
        return logits, {"att": a, "w": w, "cam": fused}

    def forward(self, x, extra=False):
        l, e = self.head(self.enc(x)); return (l, e) if extra else l

    def head_parameters(self):
        enc_ids = {id(p) for p in self.enc.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]


class TimmNet(nn.Module):
    """Standard timm classifier used as published baseline (own head, 5 classes)."""
    def __init__(self, name, pretrained=None):
        super().__init__()
        self.m = create_timm(name, pretrained, num_classes=N_CLS)

    def encode(self, x):
        return [self.m.forward_features(x)]

    def head(self, feats):
        return self.m.forward_head(feats[0]), {"cam": feats[0]}

    def forward(self, x, extra=False):
        l, e = self.head(self.encode(x)); return (l, e) if extra else l

    def head_parameters(self):
        return list(self.m.get_classifier().parameters())


def build_model(spec, pretrained=None):
    kind = spec["kind"]
    if kind == "la":
        return LANet(spec.get("backbone", PROPOSED_BACKBONE), pretrained)
    if kind == "base":
        return BaselineNet(spec.get("backbone", PROPOSED_BACKBONE), pretrained)
    return TimmNet(spec["backbone"], pretrained)


def count_params_flops(model):
    """params (M) and GMACs at 224x224 (FlopCounterMode counts 2 FLOPs per MAC)."""
    n = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        from torch.utils.flop_counter import FlopCounterMode
        was = model.training; model.eval()
        x = torch.randn(1, 3, HP["img"], HP["img"], device=next(model.parameters()).device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        model.train(was)
        return n, fc.get_total_flops() / 2e9
    except Exception as e:
        log("FLOP count failed:", e); return n, float("nan")


# ================================ losses / noise module =================================
def per_sample_loss(logits, y, kind, prior_log, cb_w):
    logits = logits.float()
    if kind == "la":                                         # logit-adjusted CE (tau = 1); raw logits are used at inference
        return F.cross_entropy(logits + prior_log[None], y, reduction="none")
    ce = F.cross_entropy(logits, y, reduction="none")
    if kind == "ce":
        return ce
    if kind == "focal":
        return (1 - torch.exp(-ce)) ** 2 * ce
    if kind == "cb":                                         # class-balanced (effective number of samples, beta=0.999)
        return cb_w[y] * ce
    if kind == "sce":                                        # symmetric CE (alpha=0.1, beta=1, log0=-9.21)
        p_y = torch.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        return 0.1 * ce + 1.0 * 9.2103 * (1 - p_y)
    raise ValueError(kind)


class NoiseWeighter:
    """Class-aware soft small-loss re-weighting, optionally coupled with attention focus.
       tau_c = q-quantile of the recent loss history of class c (so minority classes are judged against their own
       loss distribution, not the majority's). w = 1 if loss <= tau_c else exp(-(loss - tau_c) * scale / T), where
       scale = 1 + beta * (1 - focus_rank): high-loss samples with diffuse attention are down-weighted more than
       high-loss samples with focused attention (probably hard rather than mislabelled)."""
    def __init__(self, q, warm, temp, beta, hist=256):
        self.q, self.warm, self.temp, self.beta, self.hist = q, warm, temp, beta, hist
        self.buf = [torch.empty(0, device=DEVICE) for _ in range(N_CLS)]
        self.tau = torch.full((N_CLS,), float("inf"), device=DEVICE)

    @torch.no_grad()
    def __call__(self, loss, y, epoch, focus=None):
        l = loss.detach().float()
        for c in torch.unique(y).tolist():
            self.buf[c] = torch.cat([self.buf[c], l[y == c]])[-self.hist:]
            if self.buf[c].numel() >= 8:
                self.tau[c] = torch.quantile(self.buf[c], self.q)
        if epoch < self.warm:
            return torch.ones_like(l)
        excess = (l - self.tau[y]).clamp_min(0)
        scale = 1.0
        if focus is not None and self.beta > 0:
            z = (focus - focus.mean()) / (focus.std() + 1e-6)
            scale = 1 + self.beta * (1 - torch.sigmoid(z))
        return torch.exp(-excess * scale / self.temp)


def attention_focus(extra):
    """1 - normalised entropy of the attention weights (1 = single peak, 0 = uniform)."""
    w = extra["w"].float().flatten(1)
    return 1 - (-(w * torch.log(w + 1e-8)).sum(1) / math.log(w.size(1))).detach()


def inject_noise(y, rate, seed):
    """Symmetric label noise on the TRAIN labels only: with prob `rate` replace by a uniformly random other class."""
    rng = np.random.RandomState(1000 + seed)
    y0 = y.cpu().numpy().copy(); y1 = y0.copy()
    flip = rng.rand(len(y0)) < rate
    y1[flip] = (y0[flip] + rng.randint(1, N_CLS, flip.sum())) % N_CLS
    return torch.from_numpy(y1).long().to(DEVICE), float(flip.mean())


# ================================ evaluation ============================================
@torch.no_grad()
def predict(model, X_u8, bs=256):
    model.eval(); out = []
    for i in range(0, len(X_u8), bs):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_CUDA):
            out.append(model(prep_eval(to_float(X_u8[i:i + bs]))).float().cpu())
    return torch.cat(out)


def compute_metrics(y, p):
    L = list(range(N_CLS))
    return dict(acc=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro", labels=L, zero_division=0)),
                bal_acc=float(balanced_accuracy_score(y, p)),
                recall=recall_score(y, p, average=None, labels=L, zero_division=0).tolist(),
                precision=precision_score(y, p, average=None, labels=L, zero_division=0).tolist(),
                f1=f1_score(y, p, average=None, labels=L, zero_division=0).tolist())


# ================================ training ==============================================
def train_one(run_id, spec, data, seed, loss_type="ce", use_noise=False, noise_rate=0.0, lr_b=5e-4, lr_h=1e-3,
              save_ckpt=False, meta=None):
    set_seed(seed); t_start = time.time()
    model = build_model(spec).to(DEVICE).to(memory_format=torch.channels_last)
    n_params, gmacs = count_params_flops(model)
    ytr, y_clean = data.ytr.clone(), data.ytr.clone(); actual_noise = 0.0
    if noise_rate > 0:
        ytr, actual_noise = inject_noise(ytr, noise_rate, seed)
    is_noisy = (ytr != y_clean)
    cnt = torch.bincount(ytr, minlength=N_CLS).float().clamp_min(1)
    prior_log = torch.log(cnt / cnt.sum())
    eff = (1 - 0.999 ** cnt) / (1 - 0.999); cb_w = (1 / eff); cb_w = cb_w / cb_w.sum() * N_CLS

    head_ids = {id(p) for p in model.head_parameters()}
    pb = [p for p in model.parameters() if id(p) not in head_ids]
    ph = [p for p in model.parameters() if id(p) in head_ids]
    opt = torch.optim.AdamW([{"params": pb, "lr": lr_b}, {"params": ph, "lr": lr_h}], weight_decay=HP["wd"])
    base_lr = [lr_b, lr_h]
    bs, E = HP["bs"], HP["epochs"]; spe = data.n_train // bs; total = E * spe
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)
    nz = NoiseWeighter(HP["nz_q"], HP["nz_warm"], HP["nz_temp"], HP["nz_beta"] if spec["kind"] == "la" else 0.0) if use_noise else None
    best, bad, step, w_cl, w_no, hist = dict(f1=-1, ep=-1, sd=None), 0, 0, [0.0, 0], [0.0, 0], []
    diverged = False
    for ep in range(E):
        model.train(); perm = torch.randperm(data.n_train, device=DEVICE); tl = torch.zeros((), device=DEVICE); te0 = time.time()
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]; yb = ytr[idx]
            f = min(1.0, (step + 1) / HP["warmup_steps"]) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * f
            xb = norm(train_aug(to_float(data.Xtr[idx])))
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_CUDA):
                logits, ex = model(xb, extra=True)
            lv = per_sample_loss(logits, yb, loss_type, prior_log, cb_w)
            if nz is not None:
                w = nz(lv, yb, ep, attention_focus(ex) if "w" in ex else None)
                loss = (w * lv).sum() / w.sum().clamp_min(1e-6)
                if noise_rate > 0 and ep >= HP["nz_warm"]:
                    m = is_noisy[idx]
                    w_no[0] += float(w[m].sum()); w_no[1] += int(m.sum()); w_cl[0] += float(w[~m].sum()); w_cl[1] += int((~m).sum())
            else:
                loss = lv.mean()
            if "att" in ex:
                loss = loss + HP["lam_att"] * ex["att"].float().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); step += 1; tl += loss.detach()
        tl = float(tl) / spe
        if not math.isfinite(tl):
            log(f"  !! loss diverged at epoch {ep}"); diverged = True; break
        pv = predict(model, data.Xva).argmax(1).numpy()
        f1 = f1_score(data.yva_np, pv, average="macro", labels=list(range(N_CLS)), zero_division=0)
        hist.append(dict(ep=ep, loss=tl, val_f1=float(f1)))
        log(f"  {run_id} ep{ep + 1:02d}/{E} loss {tl:.4f} val-acc {np.mean(pv == data.yva_np):.4f} val-F1 {f1:.4f} ({time.time() - te0:.0f}s)")
        if f1 > best["f1"]:
            best.update(f1=float(f1), ep=ep, sd={k: v.detach().clone() for k, v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
            if ep + 1 >= HP["min_epochs"] and bad >= HP["patience"]:
                log("  early stop"); break
    if best["sd"] is None:
        rec = dict(run_id=run_id, failed=True, diverged=diverged, seed=seed, **(meta or {})); append_result(rec); return rec
    model.load_state_dict(best["sd"])
    pred = predict(model, data.Xte).argmax(1).numpy()
    m = compute_metrics(data.yte_np, pred)
    np.save(OUT_DIR / "preds" / f"{run_id}.npy", pred.astype(np.int8))
    if save_ckpt:
        torch.save(model.state_dict(), OUT_DIR / "ckpt" / f"{run_id}.pt")
    rec = dict(run_id=run_id, seed=seed, spec=spec, loss=loss_type, noise_module=use_noise, noise_rate=noise_rate,
               noise_actual=actual_noise, params_m=n_params, gmacs=gmacs, best_epoch=best["ep"] + 1, epochs_run=len(hist),
               val_macro_f1=best["f1"], test=m, train_s=time.time() - t_start, split_hash=data.split_hash,
               nz_w_clean=(w_cl[0] / w_cl[1]) if w_cl[1] else None, nz_w_noisy=(w_no[0] / w_no[1]) if w_no[1] else None,
               timm=timm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if USE_CUDA else "cpu", **(meta or {}))
    append_result(rec)
    log(f"DONE {run_id}: test acc {m['acc']:.4f} macroF1 {m['macro_f1']:.4f} CBB-recall {m['recall'][0]:.3f} "
        f"params {n_params:.2f}M GMACs {gmacs:.2f} ({rec['train_s'] / 60:.1f} min)")
    del model, opt, best
    if USE_CUDA:
        torch.cuda.empty_cache()
    return rec


def run_queue(jobs, data):
    """jobs: list of dict(kind=<time-estimate key>, default_s=..., run_id=..., **train_one kwargs). Skips finished runs and
       stops starting new runs when the session time budget would be exceeded (unfinished runs are simply re-run next time)."""
    done = load_records(); deferred = []
    for j in jobs:
        rid = j["run_id"]
        if rid in done and not done[rid].get("failed"):
            log(f"skip (already done): {rid}"); continue
        if not BUDGET.allows(j["kind"], j["default_s"]):
            deferred.append(rid); continue
        kw = {k: v for k, v in j.items() if k not in ("kind", "default_s")}
        rec = train_one(data=data, **kw)
        BUDGET.add(j["kind"], rec.get("train_s", j["default_s"]) if isinstance(rec, dict) else j["default_s"])
    if deferred:
        log(f"TIME BUDGET reached -> {len(deferred)} runs deferred (just re-run this script in a new session with this "
            f"session's output attached as input; finished runs are skipped):\n   " + ", ".join(deferred))
    return deferred


def benchmark_speed(spec, data, steps=25):
    """Measure training img/s on this GPU so you can verify the time plan before the long queue starts."""
    set_seed(0); m = build_model(spec, pretrained=False).to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), 1e-4); bs = HP["bs"]
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)
    for i in range(steps + 5):
        if i == 5:
            if USE_CUDA: torch.cuda.synchronize()
            t = time.time()
        idx = torch.randint(0, data.n_train, (bs,), device=DEVICE)
        xb = norm(train_aug(to_float(data.Xtr[idx])))
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_CUDA):
            loss = m(xb).float().logsumexp(1).mean()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if USE_CUDA: torch.cuda.synchronize()
    ips = steps * bs / (time.time() - t)
    log(f"SPEED {spec}: {ips:.0f} img/s  -> ~{data.n_train / ips:.0f}s/epoch, ~{data.n_train / ips * HP['epochs'] / 60:.0f} min per {HP['epochs']}-epoch run")
    del m, opt
    if USE_CUDA: torch.cuda.empty_cache()
    return ips


def agg_table(recs, ids_filter, label_fn):
    """mean +- std over seeds for the runs selected by ids_filter."""
    rows = {}
    for r in recs.values():
        if r.get("failed") or not ids_filter(r["run_id"]):
            continue
        rows.setdefault(label_fn(r), []).append(r)
    out = []
    for k, rs in rows.items():
        g = lambda f: np.array([f(r) for r in rs]) * 100
        ms = lambda a: f"{a.mean():.2f} +- {a.std(ddof=1) if len(a) > 1 else 0:.2f}"
        out.append(dict(config=k, n_seeds=len(rs), acc=ms(g(lambda r: r["test"]["acc"])), macro_f1=ms(g(lambda r: r["test"]["macro_f1"])),
                        bal_acc=ms(g(lambda r: r["test"]["bal_acc"])), CBB_recall=ms(g(lambda r: r["test"]["recall"][0])),
                        params_M=f"{rs[0]['params_m']:.2f}", GMACs=f"{rs[0]['gmacs']:.2f}"))
    return pd.DataFrame(out)


def banner(title):
    log("=" * 90); log(title)
    log(f"torch {torch.__version__} | timm {timm.__version__} | device {DEVICE} " +
        (torch.cuda.get_device_name(0) if USE_CUDA else "") + f" | SMOKE={SMOKE} | pretrained={USE_PRETRAINED}")
    log("=" * 90)


def check_pretrained(specs):
    """Fail fast (before the long image caching) if pretrained weights cannot be obtained."""
    if not USE_PRETRAINED:
        return
    for sp in specs:
        log("checking pretrained weights for", sp)
        m = build_model(sp); del m


# ============================== END OF SHARED CORE ======================================


# =====================================================================================
#  RUN 2 / 3  --  published-style BASELINES (3 seeds) + IMBALANCE-LOSS comparison (focal, class-balanced)
#  CE and logit-adjusted results for the loss comparison come from Run 1 (configs 'base' and 'L'; same code, same seeds).
#  Expected: ~8-9 h on a T4.  (Optional: attach Run 1 output as input; it is only needed for the final printed tables.)
# =====================================================================================
BASELINES = {   # name: (lr_backbone, lr_head, est. minutes per run)   -- same augmentation / epochs / schedule for all models
    "mobilenetv3_large_100": (5e-4, 1e-3, 20),
    "ghostnet_100": (5e-4, 1e-3, 20),
    "efficientnet_b0": (5e-4, 1e-3, 22),
    "mobilevit_s": (2e-4, 1e-3, 32),
    "convnext_tiny": (1e-4, 1e-3, 45),
}


def main():
    banner("RUN 2/3 : baselines + imbalance-loss comparison")
    names = ["mobilenetv3_large_100", "mobilevit_s"] if SMOKE else list(BASELINES)
    check_pretrained([dict(kind="timm", backbone=n) for n in names] + [dict(kind="base")])
    data = prepare_data()
    benchmark_speed(dict(kind="timm", backbone="convnext_tiny" if not SMOKE else "mobilenetv3_large_100"), data)
    seeds = [0] if SMOKE else [0, 1, 2]
    jobs = []
    for s in seeds:
        for n in names:
            lb, lh, mins = BASELINES[n]
            jobs.append(dict(kind=f"bl_{n}", default_s=mins * 60, run_id=f"bl_{n}__s{s}", spec=dict(kind="timm", backbone=n), seed=s,
                             loss_type="ce", lr_b=lb, lr_h=lh, meta=dict(stage="baseline", config=n)))
        for ls in (["focal"] if SMOKE else ["focal", "cb"]):                 # backbone + GAP head, loss is the only change
            jobs.append(dict(kind=f"loss_{ls}", default_s=20 * 60, run_id=f"loss_{ls}__s{s}", spec=dict(kind="base"), seed=s,
                             loss_type=ls, meta=dict(stage="loss", config=ls)))
    run_queue(jobs, data)
    recs = load_records()
    t1 = agg_table(recs, lambda i: i.startswith("bl_") or i == "abl_full__s0" or i.startswith("abl_full__"),
                   lambda r: r["config"] if r.get("stage") == "baseline" else "PROPOSED (full)")
    t2 = agg_table(recs, lambda i: i.startswith("loss_") or i.startswith("abl_base__") or i.startswith("abl_L__"),
                   lambda r: {"base": "CE", "L": "logit-adjusted"}.get(r.get("config"), r.get("config")))
    for nm, t in (("BASELINES vs PROPOSED", t1), ("IMBALANCE LOSSES (backbone + GAP head)", t2)):
        if len(t):
            print(f"\n{nm} (test, mean +- std, %)\n", t.to_string(index=False))
    t1.to_csv(OUT_DIR / "table_baselines_run2.csv", index=False); t2.to_csv(OUT_DIR / "table_losses_run2.csv", index=False)
    log("Run 2 finished. Save this version and attach its output to Run 3.")


if __name__ == "__main__":
    main()

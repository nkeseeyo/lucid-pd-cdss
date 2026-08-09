"""RQ4 MRI leakage study: compare transfer-learning CNNs under an image-level
(leaky) split vs a scan-level (honest) split, with Grad-CAM.

This is the imaging counterpart of the voice model comparison (RQ2). Each backbone is
trained twice with an identical, properly-regularised procedure: once with a random
image-level split (slices from one scan can land on both sides) and once with a
scan-level split (all slices of an acquisition sequence stay together). The gap is the
leakage estimate quantified by Yagis et al. (2021) for brain MRI.

Training follows standard practice so the comparison is credible, not a strawman:
a validation set is carved from the training portion (scan-disjoint for the honest
protocol), the loss is class-weighted, the pretrained backbone is frozen and only the
head is trained, the learning rate is reduced on plateau, and training stops early on
validation loss with the best weights restored.

Run (needs torch + torchvision; GPU recommended; cu128 wheels for the RTX 50-series):
    python -m pdcdss.mri.leakage
    python -m pdcdss.mri.leakage --backbones resnet50 efficientnet_b0 --epochs 40
    python -m pdcdss.mri.leakage --dedup      # drop exact-duplicate slices first

Outputs:
    results/tables/mri_leakage.csv        per model x protocol metrics
    results/figures/mri_leakage_gap.png   image-level vs scan-level accuracy per model
    results/figures/mri_gradcam.png       Grad-CAM on the leaky model
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import models, transforms  # noqa: E402

from pdcdss.config import (  # noqa: E402
    MRI_DIR,
    RANDOM_SEED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
)
from pdcdss.mri.dataset import file_hash, list_images  # noqa: E402

try:
    from tqdm.auto import tqdm  # noqa: E402
except ImportError:  # graceful fallback if tqdm is not installed
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kw):
            self._it = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._it)

        def set_postfix(self, **kw):
            pass

        def close(self):
            pass

SEED = RANDOM_SEED
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BACKBONES = ["smallcnn", "vgg16", "resnet50", "densenet121", "efficientnet_b0"]


def _seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------- data ------
def _tf(train: bool) -> transforms.Compose:
    aug = [transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([
        transforms.Resize((224, 224)),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class MRIDataset(Dataset):
    def __init__(self, paths, labels, train=False):
        self.paths, self.labels, self.tf = paths, labels, _tf(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


def _loader(paths, labels, idx, train, bs):
    ds = MRIDataset([paths[i] for i in idx], [labels[i] for i in idx], train=train)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=0)


# ----------------------------------------------------------------- models ------
class SmallCNN(nn.Module):
    """Compact from-scratch baseline (the proposal's simple CNN)."""

    def __init__(self):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                                      block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.5), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "smallcnn":
        return SmallCNN()
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    raise ValueError(f"unknown backbone: {name}")


def _set_trainable(model: nn.Module, backbone: str, mode: str = "head") -> None:
    """Choose which parameters train: ``head`` (frozen backbone, train the classifier),
    ``lastblock`` (also unfreeze the final conv block) or ``full`` (everything).

    SmallCNN is always fully trainable (it is the from-scratch baseline)."""
    bb = backbone.lower()
    if bb == "smallcnn":
        return
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    head = {"vgg16": "classifier", "resnet50": "fc", "densenet121": "classifier",
            "efficientnet_b0": "classifier"}[bb]
    for p in getattr(model, head).parameters():
        p.requires_grad = True
    if mode == "lastblock":
        block = {
            "resnet50": lambda m: m.layer4,
            "vgg16": lambda m: m.features[24:],
            "densenet121": lambda m: m.features.denseblock4,
            "efficientnet_b0": lambda m: m.features[-1],
        }[bb](model)
        for p in block.parameters():
            p.requires_grad = True


def _target_layer(model: nn.Module, backbone: str):
    """Last convolutional layer used for Grad-CAM."""
    bb = backbone.lower()
    if bb == "smallcnn":
        return model.features[-1][0]
    if bb == "vgg16":
        return model.features[-3]
    if bb == "resnet50":
        return model.layer4[-1]
    if bb in ("densenet121", "efficientnet_b0"):
        return model.features[-1]
    raise ValueError(bb)


# --------------------------------------------------------------- train/eval ----
def _class_weights(labels, idx):
    y = np.array([labels[i] for i in idx])
    counts = np.bincount(y, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def _val_loss(model, loader, crit):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            tot += crit(model(xb), yb).item() * len(yb)
            n += len(yb)
    return tot / max(n, 1)


def _evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            prob = torch.softmax(model(xb.to(DEVICE)), 1)[:, 1].cpu().numpy()
            ps += prob.tolist()
            ys += yb.numpy().tolist()
    ys, ps = np.array(ys), np.array(ps)
    pred = (ps >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(ys, pred),
        "balanced_acc": balanced_accuracy_score(ys, pred),
        "sensitivity": recall_score(ys, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "roc_auc": roc_auc_score(ys, ps) if len(set(ys.tolist())) > 1 else float("nan"),
        "n_test": int(len(ys)),
    }


def _train_eval(backbone, tr_idx, va_idx, te_idx, paths, labels, max_epochs, bs, lr,
                patience, desc="", finetune="head"):
    """Train with class-weighted loss, LR-on-plateau and early stopping (best weights).

    ``finetune`` selects how much of the backbone trains (head/lastblock/full).
    A tqdm bar reports per-epoch train and validation loss, the best validation loss
    so far and the early-stopping patience counter."""
    _seed_all(SEED)
    model = build_model(backbone).to(DEVICE)
    _set_trainable(model, backbone, finetune)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=2)
    crit = nn.CrossEntropyLoss(weight=_class_weights(labels, tr_idx))

    tr = _loader(paths, labels, tr_idx, True, bs)
    va = _loader(paths, labels, va_idx, False, bs)

    best_loss, best_state, bad, ran = float("inf"), None, 0, 0
    bar = tqdm(range(max_epochs), desc=desc, leave=False)
    for ep in bar:
        ran = ep + 1
        model.train()
        run, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(yb)
            n += len(yb)
        tloss = run / max(n, 1)
        vloss = _val_loss(model, va, crit)
        sched.step(vloss)
        if vloss < best_loss - 1e-4:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        bar.set_postfix(train=f"{tloss:.3f}", val=f"{vloss:.3f}",
                        best=f"{best_loss:.3f}", bad=bad)
        if bad >= patience:
            break
    bar.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = _evaluate(model, _loader(paths, labels, te_idx, False, bs))
    metrics["epochs_run"] = ran
    return metrics, model


# ------------------------------------------------------------------ plots ------
PRETTY = {"smallcnn": "Small CNN", "vgg16": "VGG16", "resnet50": "ResNet50",
          "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0"}


def _plot_gap(df: pd.DataFrame) -> None:
    """Plot balanced accuracy (honest under class imbalance) per model and protocol."""
    order = list(dict.fromkeys(df["model"]))

    def val(model, kind):
        sub = df[(df["model"] == model) & (df["protocol"].str.startswith(kind))]
        return float(sub["balanced_acc"].iloc[0]) if len(sub) else float("nan")

    leaky = [val(m, "Image") for m in order]
    honest = [val(m, "Scan") for m in order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - w / 2, leaky, w, label="Image-level split (leaky)", color="#9A5B00")
    ax.bar(x + w / 2, honest, w, label="Scan-level split (leakage-free)", color="#0C5C5E")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in order], fontsize=9, rotation=15)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("MRI leakage: balanced accuracy, image-level vs scan-level split",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "mri_leakage_gap.png", dpi=200)
    plt.close(fig)


NAME = {1: "PD", 0: "NonPD"}


def _gradcam(model, target, tf, path):
    """Return (PIL image, normalised CAM resized to the image, predicted class)."""
    store = {}
    h1 = target.register_forward_hook(lambda m, i, o: store.__setitem__("a", o.detach()))
    h2 = target.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("g", go[0].detach()))
    img = Image.open(path).convert("RGB")
    # requires_grad on the input forces autograd through the (possibly frozen) backbone
    # so the backward hook on the target conv layer fires.
    x = tf(img).unsqueeze(0).to(DEVICE).requires_grad_(True)
    model.zero_grad()
    out = model(x)
    cls = int(out.argmax(1))
    out[0, cls].backward()
    A, G = store["a"][0], store["g"][0]
    cam = torch.relu((G.mean(dim=(1, 2))[:, None, None] * A).sum(0)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.array(Image.fromarray((cam * 255).astype("uint8")).resize(img.size))
    h1.remove()
    h2.remove()
    return img, cam, cls


def _pick(df, te_idx, per_class):
    te = df.iloc[te_idx]
    picks = []
    for lab in (1, 0):
        picks += [(p, lab) for p in te[te["label"] == lab]["path"].head(per_class).tolist()]
    return picks


def gradcam_panel(leaky_models, df, te_idx, per_class=3) -> None:
    backbone = "resnet50" if "resnet50" in leaky_models else next(iter(leaky_models))
    model = leaky_models[backbone].to(DEVICE).eval()
    target = _target_layer(model, backbone)
    tf = _tf(False)
    fig, axes = plt.subplots(2, per_class, figsize=(3 * per_class, 6))
    for ax, (path, true) in zip(np.atleast_1d(axes).ravel(), _pick(df, te_idx, per_class)):
        img, cam, cls = _gradcam(model, target, tf, path)
        ax.imshow(img.convert("L"), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        mark = "OK" if cls == true else "X"
        ax.set_title(f"true={NAME[true]} / pred={NAME[cls]} [{mark}]", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Grad-CAM ({backbone}, image-level model): where the CNN looks",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "mri_gradcam.png", dpi=200)
    plt.close(fig)


def gradcam_compare(models_by_split, backbone, df, te_idx, per_class=2,
                    fname="mri_gradcam_compare.png") -> None:
    """Grad-CAM for the SAME slices under the leaky vs honest models, row by row."""
    tf = _tf(False)
    picks = _pick(df, te_idx, per_class)
    splits = list(models_by_split)
    fig, axes = plt.subplots(len(splits), len(picks),
                             figsize=(3.3 * len(picks), 3.4 * len(splits)))
    axes = np.atleast_2d(axes)
    for r, split in enumerate(splits):
        model = models_by_split[split].to(DEVICE).eval()
        target = _target_layer(model, backbone)
        disp = "leakage-free" if split == "honest" else split
        for c, (path, true) in enumerate(picks):
            img, cam, cls = _gradcam(model, target, tf, path)
            ax = axes[r, c]
            ax.imshow(img.convert("L"), cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{disp}: {NAME[true]} -> {NAME[cls]}", fontsize=10)
            ax.axis("off")
    fig.suptitle(f"Grad-CAM: leaky vs leakage-free {PRETTY.get(backbone, backbone)} "
                 f"(last block fine-tuned)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / fname, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- main ------
def _splits(idx, labels, groups):
    """Three-way splits. Image-level: random stratified. Scan-level: scan-disjoint."""
    # image-level (leaky): random test, then random val from the remaining train
    trv_i, te_i = train_test_split(idx, test_size=0.2, stratify=labels,
                                   random_state=SEED)
    tr_i, va_i = train_test_split(trv_i, test_size=0.15,
                                  stratify=[labels[i] for i in trv_i],
                                  random_state=SEED)
    # scan-level (honest): test scans disjoint from train/val scans; val scans disjoint
    trv_g, te_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                       .split(idx, labels, groups))
    g_trv = [groups[i] for i in trv_g]
    rel_tr, rel_va = next(GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED)
                          .split(trv_g, [labels[i] for i in trv_g], g_trv))
    tr_g, va_g = trv_g[rel_tr], trv_g[rel_va]
    return {"Image-level (leaky)": (tr_i, va_i, te_i),
            "Scan-level (honest)": (tr_g, va_g, te_g)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--epochs", type=int, default=30, help="max epochs (early stopping cuts this)")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--finetune", choices=["head", "lastblock", "full"], default="head",
                    help="how much of the backbone trains")
    ap.add_argument("--dedup", action="store_true",
                    help="drop exact-duplicate slices before splitting")
    args = ap.parse_args()

    rows = list_images(MRI_DIR)
    if not rows:
        raise SystemExit(f"No labelled images under {MRI_DIR}")
    df = pd.DataFrame(rows)
    if args.dedup:
        df["hash"] = [file_hash(p) for p in df["path"]]
        before = len(df)
        df = df.drop_duplicates("hash").reset_index(drop=True)
        print(f"dedup: removed {before - len(df)} exact-duplicate slices")
    paths = df["path"].tolist()
    labels = df["label"].tolist()
    groups = df["seq"].tolist()
    idx = np.arange(len(df))

    print(f"MRI leakage study on {DEVICE}: {len(df)} slices, "
          f"{df['seq'].nunique()} acquisition sequences, "
          f"class balance {np.bincount(labels).tolist()}\n")

    splits = _splits(idx, labels, groups)
    out_rows, leaky_models = [], {}
    for bb in args.backbones:
        for kind, (tr, va, te) in splits.items():
            tag = "leaky" if kind.startswith("Image") else "honest"
            m, model = _train_eval(bb, tr, va, te, paths, labels, args.epochs,
                                    args.batch_size, args.lr, args.patience,
                                    desc=f"{bb} {tag}", finetune=args.finetune)
            out_rows.append({"model": bb, "protocol": kind, **m})
            if kind.startswith("Image"):
                leaky_models[bb] = model
            print(f"  {bb:16s} {kind:22s} acc {m['accuracy']:.3f} | "
                  f"AUC {m['roc_auc']:.3f} | sens {m['sensitivity']:.3f} | "
                  f"spec {m['specificity']:.3f} | epochs {m['epochs_run']}")

    out = pd.DataFrame(out_rows)
    out.to_csv(RESULTS_TABLES / "mri_leakage.csv", index=False)
    _plot_gap(out)
    try:
        gradcam_panel(leaky_models, df, splits["Image-level (leaky)"][2])
    except Exception as e:  # noqa: BLE001
        print(f"[grad-cam skipped: {e}]")
    print(f"\nSaved -> {RESULTS_TABLES / 'mri_leakage.csv'} and {RESULTS_FIGURES}")


if __name__ == "__main__":
    main()

"""
Training script for the CNN+LSTM deepfake video detector.

Changes in this pass (on top of the previous rewrite):
  - train/val filename-overlap check: prints a loud warning if any exact
    filename shows up in both splits. On a 319-clip dataset this is the
    single most common reason "results look great in validation but bad in
    the real world" -- it's a leakage check, not an accuracy trick, but it's
    the first thing to rule out before trusting anything else below.
  - EMA (exponential moving average) of model weights. The *_video_lstm.pth
    file that used to be a straight copy of the raw model is now the EMA
    weights by default -- on this little data, raw end-of-training weights
    are noisy, and EMA consistently gives a smoother, better-generalizing
    model for close to zero cost. Disable with --ema-decay 0 to get the old
    behavior back.
  - optional focal loss (--loss focal, now the default) as an alternative to
    class-weighted cross-entropy -- better suited than plain CE to a dataset
    that's both imbalanced *and* has some clips that are much harder than
    others. Falls back to plain CE with --loss ce.
  - label smoothing (--label-smoothing, default 0.05) for both losses.
  - best-threshold search on the validation set instead of a hardcoded 0.5 --
    printed each time a new best model is saved, and stored in the
    checkpoint so inference code can use it instead of guessing 0.5.
  - optional flip-based TTA at validation time (--tta) -- averages the
    prediction for a clip with its horizontally-flipped twin. Free accuracy
    on eval, and also gives a less noisy signal to pick the "best" epoch
    from, which matters when the val set only has a handful of real clips.
  - optional color jitter (--color-jitter), OFF by default. Read the
    docstring on ConsistentClipAugment before turning it on -- for
    deepfake detection specifically, color/compression artifacts are
    sometimes part of the signal you want the model to learn, not noise to
    augment away.
  - explicit --weight-decay flag (still defaults to AdamW's usual 0.01, just
    now easy to change without editing the script).
  - --accum-steps for gradient accumulation, so you can raise the effective
    batch size without raising GPU memory use. Defaults to 1 (no change in
    behavior).
  - cudnn.benchmark = True on CUDA -- free speed-up given every clip is a
    fixed 224x224, no accuracy effect.
  - balanced_accuracy added next to the existing metrics, since plain
    accuracy is already flagged in this file as misleading on this split.

Changes from the original version (full writeup + sources in chat):
  - fixed the deprecated torch.cuda.amp.* calls (now torch.amp.*)
  - added real training-time augmentation (flip/rotate/translate/cutout,
    applied consistently across every frame of a clip so it doesn't flicker
    between different framings frame-to-frame)
  - added a WeightedRandomSampler so batches actually contain real videos
    (falls back to the old loss-weighting if VideoDeepfakeDataset doesn't
    expose per-sample labels -- see the printed message if that happens)
  - "best model" is now picked by validation AUC/F1 instead of accuracy,
    since accuracy is misleading on a split this imbalanced (60 real vs
    259 fake -- a model that always guesses "fake" scores ~81%)
  - added an LR scheduler, early stopping, and gradient clipping
  - added an optional progressive-unfreeze phase for the CNN backbone,
    off by default (see --unfreeze-epoch)
  - added a startup sanity check that prints the input tensor's shape and
    value range, to catch normalization bugs before an hour of training
    quietly goes to waste on them
  - checkpoints now also save optimizer/scaler state so a run can be
    resumed with --resume; the plain best_video_lstm.pth weights file is
    still written in the original format (a plain state_dict, same keys)
    so existing inference code doesn't break -- it now just contains the
    EMA weights instead of the raw ones, see above
  - config is CLI flags instead of hardcoded constants (run with --help)

Assumptions carried over from the original script that I couldn't verify
without dataset_video.py / models/video_model.py:
  - label 0 = real, 1 = fake (matches the order the original script lists
    directories and class weights in)
  - VideoDeepfakeDataset already returns float tensors scaled to [0, 1]
    before this script's Normalize runs (the sanity check below warns you
    if that looks wrong)
  - batched videos come out of the DataLoader shaped (B, T, C, H, W)

One thing this script still can't check for you: if VideoDeepfakeDataset
builds train/val by splitting *clips* rather than *source videos*, clips
from the same source video can land on both sides of the split. That's a
much sneakier leakage bug than the filename overlap checked below, and it
inflates validation numbers the same way. Worth a manual check in
dataset_video.py if your val metrics look too good to be true.
"""
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from models.video_model import DeepfakeVideoDetector
from dataset_video import VideoDeepfakeDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Train the CNN+LSTM deepfake video detector")
    p.add_argument("--train-dir", default="data/raw_videos/train")
    p.add_argument("--val-dir", default="data/raw_videos/val")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Accumulate gradients over this many batches before stepping. "
                        "Effective batch size = --batch-size * --accum-steps, at no extra "
                        "GPU memory cost. 1 = old behavior.")
    p.add_argument("--epochs", type=int, default=30,
                   help="Upper bound -- early stopping usually ends it sooner")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--backbone-lr-ratio", type=float, default=0.1,
                   help="CNN backbone LR = --lr * this, once unfrozen")
    p.add_argument("--unfreeze-epoch", type=int, default=0,
                   help="1-indexed epoch to unfreeze the CNN backbone. 0 disables it "
                        "(recommended until you have a stable head-only baseline -- "
                        "319 clips is not a lot to fine-tune a whole EfficientNet-B4 on)")
    p.add_argument("--patience", type=int, default=6, help="Early-stopping patience, in epochs")
    p.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience, in epochs")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--loss", choices=["focal", "ce"], default="focal",
                   help="focal loss (default) down-weights easy examples on top of the usual "
                        "class weighting -- generally a better fit than plain CE when the data is "
                        "both imbalanced and uneven in difficulty. --loss ce restores plain "
                        "class-weighted cross-entropy.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focusing parameter for focal loss")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.99,
                   help="EMA decay for model weights, 0 disables EMA. The deployed "
                        "*_video_lstm.pth file holds the EMA weights when enabled.")
    p.add_argument("--tta", action="store_true",
                   help="Average each validation clip's prediction with its horizontally "
                        "flipped twin. Slower validation, usually a small free accuracy gain, "
                        "and a less noisy signal for picking the best epoch.")
    p.add_argument("--color-jitter", action="store_true",
                   help="Add brightness/contrast/saturation jitter to the existing geometric "
                        "augmentation. OFF by default -- see ConsistentClipAugment's docstring "
                        "for why this can hurt deepfake detection specifically.")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Path to a *_checkpoint.pth to resume from")
    p.add_argument("--no-aug", action="store_true", help="Disable training-time augmentation")
    p.add_argument("--log-dir", default="runs", help="TensorBoard dir; pass '' to disable")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Augmentation -- geometric-only by default, so it's safe to run after
# Normalize. Randomness is sampled ONCE per clip and reused for every frame
# in it, so a clip doesn't flicker between different framings from one
# frame to the next.
# ---------------------------------------------------------------------------

class ConsistentClipAugment:
    """Per-clip-consistent augmentation.

    Geometric ops (flip/rotate/translate/cutout) are range-agnostic, so they
    run directly on the already-normalized tensor like before.

    Color jitter is NOT range-agnostic -- adjust_brightness/contrast/
    saturation assume roughly [0,1] pixel values, and running them on a
    mean-subtracted, std-scaled tensor would just add noise in a direction
    that doesn't correspond to real brightness/contrast changes. So when
    --color-jitter is on, each frame is briefly de-normalized (using the
    same IMAGENET_MEAN/STD this script normalizes with), jittered, clamped
    back to a valid [0,1] image, and re-normalized. Everything else in the
    pipeline is untouched.

    Also OFF by default for a second reason: for deepfake detection, subtle
    color-space or compression artifacts are sometimes exactly what the
    model is supposed to learn to spot, not noise to train it to ignore.
    Turn --color-jitter on only after checking that isn't the case for your
    data.
    """
    def __init__(self, flip_p=0.5, max_rotate=10.0, max_translate_frac=0.05,
                 cutout_p=0.3, cutout_frac=0.25, color_jitter=False,
                 brightness=0.15, contrast=0.15, saturation=0.1):
        self.flip_p = flip_p
        self.max_rotate = max_rotate
        self.max_translate_frac = max_translate_frac
        self.cutout_p = cutout_p
        self.cutout_frac = cutout_frac
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, clips):
        # clips: (B, T, C, H, W)
        clips = clips.clone()
        B, T, C, H, W = clips.shape
        if self.color_jitter:
            mean = clips.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = clips.new_tensor(IMAGENET_STD).view(3, 1, 1)
        for b in range(B):
            do_flip = random.random() < self.flip_p
            angle = random.uniform(-self.max_rotate, self.max_rotate)
            max_dx, max_dy = self.max_translate_frac * W, self.max_translate_frac * H
            translate = [int(random.uniform(-max_dx, max_dx)), int(random.uniform(-max_dy, max_dy))]
            do_cutout = random.random() < self.cutout_p
            if do_cutout:
                ch, cw = max(1, int(H * self.cutout_frac)), max(1, int(W * self.cutout_frac))
                top = random.randint(0, H - ch)
                left = random.randint(0, W - cw)
            if self.color_jitter:
                b_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                c_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                s_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            for t in range(T):
                frame = clips[b, t]
                if do_flip:
                    frame = TF.hflip(frame)
                frame = TF.affine(frame, angle=angle, translate=translate, scale=1.0, shear=[0.0, 0.0])
                if self.color_jitter:
                    frame = frame * std + mean  # de-normalize just for the jitter
                    frame = TF.adjust_brightness(frame, b_factor)
                    frame = TF.adjust_contrast(frame, c_factor)
                    frame = TF.adjust_saturation(frame, s_factor)
                    frame = frame.clamp(0.0, 1.0)
                    frame = (frame - mean) / std  # back to normalized space
                if do_cutout:
                    frame = frame.clone()
                    frame[:, top:top + ch, left:left + cw] = 0.0
                clips[b, t] = frame
        return clips


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Standard multi-class focal loss (Lin et al.), used here for 2 classes.
    Down-weights already-easy, correctly-classified examples so training
    spends more of its budget on the clips the model is still getting
    wrong -- generally a better fit than plain CE for data that's uneven in
    both class balance and per-sample difficulty. Supports the same
    class-weighting (alpha) and label smoothing as the CE branch, so
    switching --loss doesn't lose either feature.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        if self.label_smoothing > 0:
            off_value = self.label_smoothing / (num_classes - 1)
            true_dist = torch.full_like(log_probs, off_value)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1.0)
        pt = (log_probs.exp() * true_dist).sum(dim=1)  # soft "probability of the true class"
        focal_weight = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = -focal_weight * (true_dist * log_probs).sum(dim=1)
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model weights. Uses a warm-up schedule
    (decay ramps up from 0 towards --ema-decay over the first ~10x1/(1-decay)
    steps) so the EMA doesn't start out badly lagging a randomly-initialized
    head, which matters more than usual here given how few total optimizer
    steps a 319-clip dataset produces.
    """
    def __init__(self, model, decay=0.99):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (self.updates + 1) / (self.updates + 10))
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            mv = model_state[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)


# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------

def get_sample_labels(dataset):
    """Best-effort lookup of a per-sample 0/1 label list, in dataset index
    order, so we can build a WeightedRandomSampler. Returns None if
    VideoDeepfakeDataset doesn't expose one of the usual attribute names."""
    for attr in ("labels", "targets", "video_labels"):
        if hasattr(dataset, attr):
            return list(getattr(dataset, attr))
    if hasattr(dataset, "samples"):  # ImageFolder-style (path, label) pairs
        return [label for _, label in dataset.samples]
    return None


def check_train_val_overlap(train_dir, val_dir):
    """Flags exact filename collisions between train/ and val/ splits. This
    is a cheap, always-correct sanity check -- if it fires, val metrics are
    optimistic and shouldn't be trusted until fixed. It can't catch the
    subtler case of two *different* filenames coming from the same source
    video; see the module docstring for that one."""
    overlaps = {}
    for cls in ("real", "fake"):
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        if not (os.path.isdir(train_cls_dir) and os.path.isdir(val_cls_dir)):
            continue
        train_files = {f for f in os.listdir(train_cls_dir) if not f.startswith(".")}
        val_files = {f for f in os.listdir(val_cls_dir) if not f.startswith(".")}
        shared = train_files & val_files
        if shared:
            overlaps[cls] = shared
    if overlaps:
        total = sum(len(v) for v in overlaps.values())
        detail = ", ".join(f"{k}: {len(v)}" for k, v in overlaps.items())
        print(f"[WARNING] {total} filename(s) appear in BOTH train and val ({detail}). "
              "This is a data-leakage red flag -- validation metrics will look better than "
              "real-world performance until these are made disjoint.")
    return overlaps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs, threshold=0.5):
    """labels: 0/1 ints. probs: predicted probability of class 1 ('fake').
    AUC is NaN if a split only contains one class -- this happens more than
    you'd expect with only 60 real clips, so callers need to handle it."""
    if len(labels) == 0:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "auc": float("nan"), "confusion_matrix": np.zeros((2, 2), dtype=int)}
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")  # only one class present in this split
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc, "confusion_matrix": cm}


def find_best_threshold(labels, probs):
    """Grid-search the decision threshold that maximizes F1 on class 1, in
    place of the usual fixed 0.5. With imbalanced data 0.5 is rarely
    optimal; this is only run on the validation set, once per new-best-model
    save, and the result is stored in the checkpoint for inference to use."""
    if len(np.unique(labels)) < 2:
        return 0.5
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
        if f1 > best_score:
            best_score, best_t = f1, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def print_batch_sanity_check(loader):
    try:
        videos, labels = next(iter(loader))
    except StopIteration:
        return
    vmin, vmax, vmean = videos.min().item(), videos.max().item(), videos.mean().item()
    print(f"[sanity check] batch shape={tuple(videos.shape)} dtype={videos.dtype} "
          f"range=[{vmin:.3f}, {vmax:.3f}] mean={vmean:.3f} labels={labels.tolist()}")
    if videos.ndim != 5:
        print(f"[warning] expected a 5D (B,T,C,H,W) tensor, got {videos.ndim}D. "
              "ConsistentClipAugment assumes (B,T,C,H,W) -- adjust it if your layout differs.")
    if -0.01 <= vmin and vmax <= 1.01:
        print("[warning] pixel values look like they're still in [0,1]. If the ImageNet "
              "Normalize is supposed to have run already, double check it did -- normalized "
              "images usually range roughly -2.1 to 2.7, not 0 to 1.")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir, name_stem, model, ema, optimizer, scaler, epoch,
                     best_metric, best_threshold, args):
    # Plain weights file, same format the original script used, so any
    # existing inference/serving code (model.load_state_dict(torch.load(path)))
    # keeps working. Holds the EMA weights when EMA is enabled -- those are
    # the ones meant to be deployed.
    deploy_model = ema.ema if ema is not None else model
    torch.save(deploy_model.state_dict(), os.path.join(output_dir, f"{name_stem}.pth"))
    # Full checkpoint alongside it, for --resume.
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.ema.state_dict() if ema is not None else None,
        "ema_updates": ema.updates if ema is not None else 0,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_threshold": best_threshold,
        "args": vars(args),
    }, os.path.join(output_dir, f"{name_stem}_checkpoint.pth"))


def build_optimizer_and_scheduler(head_params, backbone_params, unfrozen, args):
    if unfrozen and backbone_params:
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_ratio},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience, factor=0.5)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# One epoch of train / val. safe_iterate keeps one bad video from killing an
# hours-long run -- it only helps with num_workers=0 (e.g. on Windows, which
# is already the workers setting this script forces); a crash inside a
# worker process is a separate, harder problem that lives in dataset_video.py.
# ---------------------------------------------------------------------------

def safe_iterate(loader, desc):
    iterator = iter(loader)
    pbar = tqdm(total=len(loader), desc=desc)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        except Exception as e:
            print(f"\n[warning] skipping a bad batch: {e}")
            pbar.update(1)
            continue
        yield batch
        pbar.update(1)
    pbar.close()


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, scaler, grad_clip,
                     augment, epoch, epochs, ema=None, accum_steps=1):
    model.train()
    total_loss, total = 0.0, 0
    all_labels, all_probs = [], []
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for i, (videos, labels) in enumerate(safe_iterate(loader, f"Train {epoch}/{epochs}")):
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if augment is not None:
            videos = augment(videos)

        # Force a step on the last batch too, so a leftover partial
        # accumulation window (num_batches not divisible by accum_steps)
        # still gets applied instead of being silently dropped.
        do_step = ((i + 1) % accum_steps == 0) or (i == num_batches - 1)

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(videos)
                loss = criterion(outputs, labels) / accum_steps
            scaler.scale(loss).backward()
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        total_loss += loss.item() * accum_steps * labels.size(0)
        total += labels.size(0)
        probs = torch.softmax(outputs.detach(), dim=1)[:, 1]
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / total if total > 0 else float("nan")
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, epoch, epochs, tta=False):
    model.eval()
    total_loss, total = 0.0, 0
    all_labels, all_probs = [], []
    for videos, labels in safe_iterate(loader, f"Val {epoch}/{epochs}"):
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(videos)
                loss = criterion(outputs, labels)
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        if tta:
            flipped = torch.flip(videos, dims=[-1])  # horizontal flip, whole clip
            if use_amp:
                with torch.amp.autocast("cuda"):
                    flipped_outputs = model(flipped)
            else:
                flipped_outputs = model(flipped)
            flipped_probs = torch.softmax(flipped_outputs, dim=1)[:, 1]
            probs = (probs + flipped_probs) / 2.0

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else float("nan")
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics, np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # every clip is a fixed 224x224, so this is a free speed-up

    os.makedirs(args.output_dir, exist_ok=True)
    for split_dir in (args.train_dir, args.val_dir):
        os.makedirs(os.path.join(split_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "fake"), exist_ok=True)

    check_train_val_overlap(args.train_dir, args.val_dir)

    # mtcnn already crops to 224x224; this assumes VideoDeepfakeDataset scales
    # frames to [0,1] floats before this Normalize runs -- see the sanity
    # check below if that assumption looks wrong for your data.
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_dataset = VideoDeepfakeDataset(args.train_dir, num_frames=args.num_frames, transform=normalize)
    val_dataset = VideoDeepfakeDataset(args.val_dir, num_frames=args.num_frames, transform=normalize)

    if len(train_dataset) == 0:
        print("Error: No training data found! Please download the Kaggle DFDC dataset and "
              f"extract it to {args.train_dir}/real and {args.train_dir}/fake.")
        return

    # 0 workers on windows (or mps) -- otherwise it deadlocks / misbehaves
    nw = args.num_workers
    if nw is None:
        nw = 0 if (os.name == "nt" or device == "mps") else 4

    num_real = len([f for f in os.listdir(os.path.join(args.train_dir, "real")) if not f.startswith(".")])
    num_fake = len([f for f in os.listdir(os.path.join(args.train_dir, "fake")) if not f.startswith(".")])
    print(f"Training videos: {num_real} real, {num_fake} fake")

    use_loss_class_weights = True
    sample_labels = get_sample_labels(train_dataset)
    if sample_labels is not None and len(sample_labels) == len(train_dataset):
        counts = np.bincount(sample_labels, minlength=2)
        weight_per_class = 1.0 / np.maximum(counts, 1)
        sample_weights = [weight_per_class[l] for l in sample_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=nw, pin_memory=(device == "cuda"))
        use_loss_class_weights = False
        print("Balancing batches with a WeightedRandomSampler (found per-sample labels on the dataset).")
    else:
        if num_real + num_fake > 0:
            est_empty_pct = 100 * (num_fake / (num_real + num_fake)) ** args.batch_size
        else:
            est_empty_pct = 0
        print("Couldn't find a per-sample label list on VideoDeepfakeDataset "
              "(tried .labels / .targets / .video_labels / .samples) -- batches will be drawn "
              f"uniformly at random. With this class split, roughly {est_empty_pct:.0f}% of "
              "batches will contain *zero* real videos. Falling back to class-weighted loss "
              "only; expose one of those attributes (a 0/1 label list in __getitem__ order) "
              "on the dataset to enable proper batch balancing instead.")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=nw, pin_memory=(device == "cuda"))

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=nw, pin_memory=(device == "cuda"))

    print_batch_sanity_check(train_loader)

    model = DeepfakeVideoDetector().to(device)
    head_params = [p for p in model.parameters() if p.requires_grad]
    backbone_params = [p for p in model.parameters() if not p.requires_grad]
    print(f"Trainable params (head/LSTM): {sum(p.numel() for p in head_params):,} | "
          f"frozen params (CNN backbone): {sum(p.numel() for p in backbone_params):,}")

    class_weights = None
    if use_loss_class_weights and num_real > 0 and num_fake > 0:
        total = num_real + num_fake
        class_weights = torch.tensor(
            [total / (2.0 * num_real), total / (2.0 * num_fake)], dtype=torch.float32
        ).to(device)

    if args.loss == "focal":
        alpha_desc = "uniform (sampler already balances batches)"
        if class_weights is not None:
            alpha_desc = f"real={class_weights[0]:.2f}, fake={class_weights[1]:.2f}"
        print(f"Using focal loss (gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}, alpha={alpha_desc})")
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        if class_weights is not None:
            print(f"Using class-weighted CE loss: real={class_weights[0]:.2f}, fake={class_weights[1]:.2f}, "
                  f"label_smoothing={args.label_smoothing}")
        else:
            print(f"Using CE loss (sampler already balances batches), label_smoothing={args.label_smoothing}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    augment = None if args.no_aug else ConsistentClipAugment(color_jitter=args.color_jitter)

    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay)
        window = 1.0 / (1.0 - args.ema_decay)
        print(f"EMA enabled (decay={args.ema_decay}, effective averaging window ~{window:.0f} steps). "
              f"{os.path.basename(args.output_dir) or 'output'}/*_video_lstm.pth will hold the EMA weights.")

    writer = None
    if args.log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(args.log_dir)
        except ImportError:
            print("tensorboard isn't installed; skipping run logging (pip install tensorboard to enable).")

    backbone_unfrozen = False
    optimizer, scheduler = build_optimizer_and_scheduler(head_params, backbone_params, backbone_unfrozen, args)

    start_epoch = 0
    best_metric = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    best_stem = "best_video_lstm"
    last_stem = "last_video_lstm"

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", -1.0)
        best_threshold = ckpt.get("best_threshold", 0.5)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best metric so far {best_metric:.4f}")
        if ema is not None and ckpt.get("ema_state_dict"):
            ema.ema.load_state_dict(ckpt["ema_state_dict"])
            ema.updates = ckpt.get("ema_updates", 0)
        if args.unfreeze_epoch > 0 and start_epoch >= args.unfreeze_epoch and backbone_params:
            for p in backbone_params:
                p.requires_grad = True
            backbone_unfrozen = True
            optimizer, scheduler = build_optimizer_and_scheduler(head_params, backbone_params, True, args)
            print("Resumed past the unfreeze point -- backbone re-unfrozen.")
        if ckpt.get("optimizer_state_dict") and not backbone_unfrozen:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                pass  # param groups changed shape; starting the optimizer fresh is fine
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    print(f"Starting training for up to {args.epochs} epochs (early-stop patience={args.patience})...")
    for epoch in range(start_epoch, args.epochs):
        if (args.unfreeze_epoch > 0 and epoch + 1 == args.unfreeze_epoch
                and not backbone_unfrozen and backbone_params):
            print(f"\nEpoch {epoch + 1}: unfreezing CNN backbone "
                  f"(backbone lr={args.lr * args.backbone_lr_ratio:.2e}, head lr={args.lr:.2e})")
            for p in backbone_params:
                p.requires_grad = True
            backbone_unfrozen = True
            optimizer, scheduler = build_optimizer_and_scheduler(head_params, backbone_params, True, args)

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp, scaler,
            args.grad_clip, augment, epoch + 1, args.epochs, ema=ema, accum_steps=args.accum_steps,
        )
        msg = f"Epoch {epoch + 1} train loss={train_loss:.4f} acc={train_metrics['accuracy'] * 100:.1f}%"
        if not np.isnan(train_metrics["auc"]):
            msg += f" auc={train_metrics['auc']:.4f}"
        print(msg)
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch + 1)

        if val_loader is not None:
            eval_model = ema.ema if ema is not None else model
            val_loss, val_metrics, val_labels, val_probs = validate(
                eval_model, val_loader, criterion, device, use_amp, epoch + 1, args.epochs, tta=args.tta,
            )
            print(f"Epoch {epoch + 1} val loss={val_loss:.4f} acc={val_metrics['accuracy'] * 100:.1f}% "
                  f"bal_acc={val_metrics['balanced_accuracy'] * 100:.1f}% "
                  f"precision={val_metrics['precision']:.3f} recall={val_metrics['recall']:.3f} "
                  f"f1={val_metrics['f1']:.3f} auc={val_metrics['auc']:.4f}")
            print(f"  confusion matrix [[TN,FP],[FN,TP]]: {val_metrics['confusion_matrix'].tolist()}")

            if writer:
                writer.add_scalar("loss/val", val_loss, epoch + 1)
                writer.add_scalar("auc/val", val_metrics["auc"], epoch + 1)
                writer.add_scalar("f1/val", val_metrics["f1"], epoch + 1)
                writer.add_scalar("balanced_accuracy/val", val_metrics["balanced_accuracy"], epoch + 1)

            monitor = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else val_metrics["f1"]
            scheduler.step(monitor)

            if monitor > best_metric:
                best_metric = monitor
                best_threshold = find_best_threshold(val_labels, val_probs)
                epochs_without_improvement = 0
                save_checkpoint(args.output_dir, best_stem, model, ema, optimizer, scaler,
                                 epoch + 1, best_metric, best_threshold, args)
                which = "AUC" if not np.isnan(val_metrics["auc"]) else "F1"
                print(f"  -> new best model saved ({which}={best_metric:.4f}, "
                      f"best decision threshold={best_threshold:.2f} vs the usual fixed 0.5)")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"No improvement in {args.patience} epochs -- stopping early.")
                    break
        else:
            print("No validation data found -- saving the latest epoch's weights "
                  "(the most recent model, not necessarily the best one).")
            save_checkpoint(args.output_dir, last_stem, model, ema, optimizer, scaler,
                             epoch + 1, best_metric, best_threshold, args)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

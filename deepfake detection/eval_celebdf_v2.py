
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

try:
    import timm  # type: ignore

    HAS_TIMM = True
except Exception:
    timm = None  # type: ignore
    HAS_TIMM = False


try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_import_error = exc

try:
    from facenet_pytorch import MTCNN  # type: ignore

    HAS_MTCNN = True
except Exception:
    HAS_MTCNN = False


class SimpleFaceCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleFaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self.classifier(self.features(x))


def detect_arch(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    # EfficientNet checkpoints (torchvision) have MBConv blocks under `features.*.block.*`
    if any(k.startswith("features.") and ".block." in k for k in keys):
        return "efficientnet_b0"
    if "conv1.weight" in keys or any(k.startswith("layer1.") for k in keys):
        return "resnet18"
    if any(k.startswith("features.") for k in keys):
        return "deepfacecnn"
    return "unknown"


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """Strip common training wrappers/prefixes (e.g., DataParallel module.)."""
    if not state_dict:
        return state_dict

    prefixes = ("module.", "model.", "net.", "backbone.")
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def infer_num_classes_from_state_dict(state_dict: dict, arch_hint: Optional[str] = None) -> Optional[int]:
    # Prefer unambiguous "final head" keys first.
    for k in (
        "classifier.4.weight",   # SimpleFaceCNN final linear
        "fc.weight",             # ResNet
        "classifier.weight",     # some timm models
        "head.weight",           # some timm models
        "head.fc.weight",        # some timm models
    ):
        w = state_dict.get(k)
        shape = getattr(w, "shape", None)
        if shape is not None and len(shape) == 2 and int(shape[0]) > 0:
            return int(shape[0])

    # EfficientNet (torchvision) final layer is classifier.1.weight
    # BUT SimpleFaceCNN also has classifier.1.weight (=512), so only trust this when arch is efficientnet.
    if arch_hint == "efficientnet_b0":
        w = state_dict.get("classifier.1.weight")
        shape = getattr(w, "shape", None)
        if shape is not None and len(shape) == 2 and int(shape[0]) > 0:
            return int(shape[0])

    # Fallback: pick the smallest plausible head out_features
    best: Optional[int] = None
    for k, w in state_dict.items():
        if not k.endswith(".weight"):
            continue
        if k == "classifier.1.weight" and arch_hint != "efficientnet_b0":
            continue
        if not (
            k == "fc.weight"
            or k.endswith(".fc.weight")
            or "classifier" in k
            or k.startswith("head.")
            or ".head." in k
        ):
            continue
        shape = getattr(w, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        out_features = int(shape[0])
        if out_features <= 0:
            continue
        if best is None or out_features < best:
            best = out_features
    return best


def get_transform(arch: str, img_size: int):
    if arch in ("resnet18", "xception", "efficientnet_b0"):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )





def preprocess_image(
    img: Image.Image,
    *,
    arch: str,
    img_size: int,
    tv_tfm,  
) -> torch.Tensor:
    return tv_tfm(img.convert("RGB"))


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        # torchvision EfficientNet classifier: Sequential(Dropout, Linear)
        if (
            hasattr(m, "classifier")
            and isinstance(m.classifier, nn.Sequential)
            and len(m.classifier) >= 2
            and isinstance(m.classifier[1], nn.Linear)
        ):
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        else:
            raise RuntimeError("Unexpected efficientnet_b0 classifier structure.")
        return m
    if arch == "xception":
        if not HAS_TIMM:
            raise RuntimeError("timm is required for xception. Install: pip install timm")
        # timm will set the correct classifier head when num_classes is provided
        return timm.create_model("xception", pretrained=False, num_classes=num_classes)
    if arch == "deepfacecnn":
        return SimpleFaceCNN(num_classes=num_classes)
    raise RuntimeError(f"Unknown arch: {arch}")


def load_state_dict(ckpt_path: str, device: torch.device) -> dict:
    # Compatibility with new torch versions that may accept weights_only
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint format unexpected (dict bekleniyordu).")

    # Some checkpoints save {'state_dict': ...}
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return normalize_state_dict_keys(state["state_dict"])

    return normalize_state_dict_keys(state)


def read_list_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def find_celebdf_video_roots(root: str) -> Tuple[List[str], List[str]]:
    """Returns (real_dirs, fake_dirs) candidates."""
    # Typical Celeb-DF(v2) structure
    candidates_real = [
        os.path.join(root, "Celeb-real"),
        os.path.join(root, "YouTube-real"),
    ]
    candidates_fake = [
        os.path.join(root, "Celeb-synthesis"),
    ]
    real_dirs = [d for d in candidates_real if os.path.isdir(d)]
    fake_dirs = [d for d in candidates_fake if os.path.isdir(d)]

    # If user points directly at a folder that contains mp4s
    if not real_dirs and not fake_dirs:
        mp4s = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        if mp4s:
            # unknown labels, treat as "real" list empty and "fake" empty
            return [], []

    return real_dirs, fake_dirs


def list_videos_in_dir(video_dir: str) -> List[str]:
    if not os.path.isdir(video_dir):
        return []
    out: List[str] = []
    for f in os.listdir(video_dir):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            out.append(os.path.join(video_dir, f))
    return sorted(out)


def sample_frames(video_path: str, frames_per_video: int) -> List[Image.Image]:
    if cv2 is None:
        raise RuntimeError(
            "opencv-python import edilemedi. Kurulum: pip install opencv-python\n"
            f"Detay: {_cv2_import_error}"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
    images: List[Image.Image] = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        # BGR -> RGB
        frame_rgb = frame[:, :, ::-1]
        img = Image.fromarray(frame_rgb)
        images.append(img)

    cap.release()
    return images


def mtcnn_crop_largest(img: Image.Image, mtcnn: "MTCNN", out_size: int) -> Optional[Image.Image]:
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box = boxes[int(np.argmax(areas))]
    x1, y1, x2, y2 = [int(v) for v in box]

    # clamp
    w, h = img.size
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return img.resize((out_size, out_size))

    face = img.crop((x1, y1, x2, y2)).resize((out_size, out_size))
    return face


def parse_class_names(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    if os.path.isfile(arg):
        names = read_list_file(arg)
        return names if names else None
    # comma-separated
    names = [x.strip() for x in arg.split(",") if x.strip()]
    return names if names else None


def write_csv(path: str, rows: List[dict]):
    import csv

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    out["accuracy"] = float((y_true == y_pred).mean()) if y_true.size else float("nan")

    # Optional sklearn
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        if len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        pass

    return out


@dataclass
class VideoItem:
    path: str
    label: int  # 0 real, 1 fake


def main():
    ap = argparse.ArgumentParser(description="Evaluate a .pth model on Celeb-DF(v2) videos")
    ap.add_argument("--celebdf_root", required=True, help="Celeb-DF(v2) root folder")
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--arch",
        default="auto",
        choices=["auto", "resnet18", "efficientnet_b0", "deepfacecnn", "xception"],
        help="Model architecture. Use 'auto' to infer from checkpoint.",
    )
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_per_video", type=int, default=20)
    ap.add_argument(
        "--class_names",
        default=None,
        help="Comma-separated class names OR a txt file. For multi-class models include 'original'.",
    )
    ap.add_argument(
        "--original_class",
        default="original",
        help="Name of the real/original class used for fake_score=1-p(original).",
    )
    ap.add_argument(
        "--original_index",
        type=int,
        default=-1,
        help="Explicit index of the real/original class in the model output (overrides --original_class).",
    )
    ap.add_argument(
        "--auto_original_index",
        action="store_true",
        help="Try each class index as 'original' and pick the best ROC-AUC (requires sklearn).",
    )
    ap.add_argument(
        "--force_binary",
        action="store_true",
        help="Evaluate as binary: fake_prob uses class index 1 if num_classes==2.",
    )
    ap.add_argument(
        "--use_mtcnn",
        action="store_true",
        help="Use MTCNN face crop (requires facenet-pytorch).",
    )
    ap.add_argument("--output_csv", default="celebdf_results.csv")
    ap.add_argument("--max_videos", type=int, default=0, help="0 means all")
    ap.add_argument(
        "--balance",
        action="store_true",
        help="Balance real/fake by truncating to the smaller class (old behavior).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for fake_score -> y_pred.",
    )
    ap.add_argument(
        "--auto_threshold",
        action="store_true",
        help="Choose threshold that maximizes accuracy on the evaluated set.",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle videos before balancing/truncation (reduces filename-order bias).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--list_file",
        default=None,
        help="Optional txt file listing test video filenames (one per line).",
    )

    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    celeb_root = args.celebdf_root
    ckpt_path = args.ckpt

    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(2)

    if not os.path.isdir(celeb_root):
        print(f"Celeb-DF root not found: {celeb_root}")
        sys.exit(2)

    device = torch.device(args.device)

    state = load_state_dict(ckpt_path, device)
    inferred_num_classes = infer_num_classes_from_state_dict(state)

    class_names = parse_class_names(args.class_names)
    if class_names is None:
        if inferred_num_classes == 2:
            class_names = ["original", "fake"]
        elif inferred_num_classes == 7:
            # Sensible defaults; works best if your ckpt was trained with this order.
            class_names = [
                "DeepFakeDetection",
                "Deepfakes",
                "Face2Face",
                "FaceShifter",
                "FaceSwap",
                "NeuralTextures",
                "original",
            ]
        elif inferred_num_classes is not None and inferred_num_classes > 0:
            class_names = [f"class{i}" for i in range(int(inferred_num_classes))]
        else:
            class_names = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "original"]

    if inferred_num_classes is not None and len(class_names) != int(inferred_num_classes):
        print(
            "class_names count does not match checkpoint head.\n"
            f"- class_names: {len(class_names)}\n"
            f"- checkpoint inferred num_classes: {int(inferred_num_classes)}\n"
            "Fix: pass matching --class_names (comma-separated or txt file)."
        )
        sys.exit(2)

    num_classes = len(class_names)
    if args.arch != "auto":
        arch = str(args.arch)
    else:
        arch = detect_arch(state)
        if arch == "unknown":
            print(
                "Checkpoint arch could not be detected. "
                "Please pass --arch (e.g., --arch xception)."
            )
            sys.exit(2)

    if arch == "xception" and not HAS_TIMM:
        print("timm is required for xception. Install: pip install timm")
        sys.exit(2)

    model = build_model(arch, num_classes=num_classes)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tfm = get_transform(arch, args.img_size)
    

    if args.use_mtcnn and not HAS_MTCNN:
        print("facenet-pytorch not available; run: pip install facenet-pytorch")
        sys.exit(2)

    mtcnn = None
    if args.use_mtcnn and HAS_MTCNN:
        mtcnn = MTCNN(keep_all=True, device=device, image_size=args.img_size, margin=20)

    real_dirs, fake_dirs = find_celebdf_video_roots(celeb_root)

    # Load optional list file (names only)
    allowed_names = None
    if args.list_file:
        if not os.path.isfile(args.list_file):
            print(f"List file not found: {args.list_file}")
            sys.exit(2)
        allowed_names = set(os.path.basename(x) for x in read_list_file(args.list_file))

    real_items: List[VideoItem] = []
    fake_items: List[VideoItem] = []

    # If structured dirs exist, use them
    for d in real_dirs:
        for vp in list_videos_in_dir(d):
            if allowed_names and os.path.basename(vp) not in allowed_names:
                continue
            real_items.append(VideoItem(path=vp, label=0))

    for d in fake_dirs:
        for vp in list_videos_in_dir(d):
            if allowed_names and os.path.basename(vp) not in allowed_names:
                continue
            fake_items.append(VideoItem(path=vp, label=1))

    if not real_items and not fake_items:
        # Fallback: scan root recursively, but cannot infer label
        print(
            "No standard Celeb-DF folders found (Celeb-real/YouTube-real/Celeb-synthesis).\n"
            "Please pass a correct --celebdf_root, or use the standard dataset structure."
        )
        sys.exit(2)

    if args.balance:
        n = min(len(real_items), len(fake_items))
        if n == 0:
            print(f"Cannot balance: real={len(real_items)} fake={len(fake_items)}")
            sys.exit(2)
        real_items = real_items[:n]
        fake_items = fake_items[:n]

    # Varsayılan: klasördeki TÜM videoları değerlendir (dengelemeye zorlamaz)
    items: List[VideoItem] = real_items + fake_items

    if args.shuffle:
        rng.shuffle(items)

    n_real = sum(1 for it in items if it.label == 0)
    n_fake = sum(1 for it in items if it.label == 1)
    if args.balance:
        print(f"Balanced set: real={n_real} fake={n_fake} total={len(items)}")
    else:
        print(f"All videos: real={n_real} fake={n_fake} total={len(items)}")

    # max_videos verilirse: sadece ilk N video (balance modunda çift sayıya indir)
    if args.max_videos and args.max_videos > 0:
        max_total = min(int(args.max_videos), len(items))
        if args.balance and (max_total % 2 == 1):
            max_total -= 1
        items = items[:max_total]
        n_real = sum(1 for it in items if it.label == 0)
        n_fake = sum(1 for it in items if it.label == 1)
        print(f"After max_videos: total={len(items)} (real={n_real}, fake={n_fake})")

    # Determine original index if multi-class
    orig_idx: Optional[int] = None
    if args.original_index is not None and int(args.original_index) >= 0:
        if int(args.original_index) >= num_classes:
            print(f"original_index out of range: {args.original_index} (num_classes={num_classes})")
            sys.exit(2)
        orig_idx = int(args.original_index)
    else:
        target = (args.original_class or "").strip().lower()
        if target:
            for i, nm in enumerate(class_names):
                if nm.strip().lower() == target:
                    orig_idx = i
                    break

    if orig_idx is None and not args.force_binary:
        print(
            "Warning: could not resolve original class index from class_names. "
            "If class order is mismatched, pass --original_index or enable --auto_original_index."
        )

    rows: List[dict] = []
    y_true: List[int] = []
    mean_probs_all: List[np.ndarray] = []

    print(f"Arch: {arch} | num_classes: {num_classes} | device: {device}")
    print(f"Videos to evaluate: {len(items)}")

    for idx, it in enumerate(items, start=1):
        frames = sample_frames(it.path, frames_per_video=args.frames_per_video)
        if not frames:
            # Skip unreadable videos
            continue

        per_frame_probs: List[np.ndarray] = []
        pred_names = []

        for fr in frames:
            img = fr.convert("RGB")
            if mtcnn is not None:
                crop = mtcnn_crop_largest(img, mtcnn, out_size=args.img_size)
                if crop is None:
                    continue
                img = crop

            x0 = preprocess_image(
                img,
                arch=arch,
                img_size=args.img_size,
                tv_tfm=tfm,
                
            )
            x = x0.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

            per_frame_probs.append(probs)

            pred_idx = int(np.argmax(probs))
            pred_names.append(class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx))

        if not per_frame_probs:
            continue

        mean_probs = np.mean(np.stack(per_frame_probs, axis=0), axis=0).astype(np.float32)
        pred_name_mode = max(set(pred_names), key=pred_names.count)

        rows.append(
            {
                "video": it.path,
                "y_true": it.label,
                "fake_score": None,
                "y_pred": None,
                "n_frames": int(len(per_frame_probs)),
                "pred_name_mode": pred_name_mode,
            }
        )

        y_true.append(it.label)
        mean_probs_all.append(mean_probs)

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(items)}")

    if not rows:
        print("No results produced (videos unreadable or zero frames).")
        sys.exit(2)

    y_true_np = np.array(y_true, dtype=np.int64)
    probs_np = np.stack(mean_probs_all, axis=0)

    chosen_orig_idx: Optional[int]
    if args.force_binary and num_classes == 2:
        chosen_orig_idx = None
    else:
        chosen_orig_idx = orig_idx

    if args.auto_original_index and (not (args.force_binary and num_classes == 2)) and num_classes > 1:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore

            best_auc = -1.0
            best_i: Optional[int] = None
            if len(np.unique(y_true_np)) == 2:
                for i in range(num_classes):
                    scores_i = 1.0 - probs_np[:, i]
                    auc_i = float(roc_auc_score(y_true_np, scores_i))
                    if auc_i > best_auc:
                        best_auc = auc_i
                        best_i = int(i)
            if best_i is not None:
                chosen_orig_idx = best_i
                print(
                    f"Auto-selected original_index={chosen_orig_idx} ({class_names[chosen_orig_idx]}) with roc_auc={best_auc:.4f}"
                )
        except Exception as exc:
            print(f"auto_original_index requested but failed (sklearn needed). Details: {exc}")

    if (chosen_orig_idx is None) and (not (args.force_binary and num_classes == 2)):
        print(
            "Warning: original class index unresolved; fake_score falls back to probs[1]. "
            "This is likely wrong if your class order doesn't match the checkpoint."
        )

    # Build y_score from per-video mean probs
    if args.force_binary and num_classes == 2:
        y_score_np = probs_np[:, 1].astype(np.float32)
    else:
        if chosen_orig_idx is not None and 0 <= int(chosen_orig_idx) < num_classes:
            y_score_np = (1.0 - probs_np[:, int(chosen_orig_idx)]).astype(np.float32)
        else:
            y_score_np = (probs_np[:, 1] if num_classes > 1 else np.zeros(len(rows))).astype(
                np.float32
            )

    # Thresholding
    thr = float(args.threshold)
    if args.auto_threshold:
        grid = np.linspace(0.0, 1.0, 201)
        best_thr = thr
        best_acc = -1.0
        for t in grid:
            pred_t = (y_score_np >= t).astype(np.int64)
            acc_t = float((pred_t == y_true_np).mean()) if y_true_np.size else float("nan")
            if acc_t > best_acc:
                best_acc = acc_t
                best_thr = float(t)
        thr = best_thr
        print(f"Auto-selected threshold={thr:.3f} with accuracy={best_acc:.4f}")

    y_pred_np = (y_score_np >= thr).astype(np.int64)

    for r, s, p in zip(rows, y_score_np.tolist(), y_pred_np.tolist()):
        r["fake_score"] = float(s)
        r["y_pred"] = int(p)

    write_csv(args.output_csv, rows)

    metrics = try_metrics(y_true_np, y_score_np, y_pred_np)

    print("\nSaved:", args.output_csv)
    if (chosen_orig_idx is not None) and (not (args.force_binary and num_classes == 2)):
        print(f"original_index: {chosen_orig_idx} | original_class: {class_names[int(chosen_orig_idx)]}")
    print(f"threshold: {thr:.3f}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()

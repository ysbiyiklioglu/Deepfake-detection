import os
import glob
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

import tempfile

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except Exception:
    HAS_MTCNN = False




# ---- Models ----
class SimpleFaceCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def detect_arch(state_dict: dict) -> str:
    keys = list(state_dict.keys())

    # torchvision EfficientNet-B0 tipik imza: features.*.block.* veya features.0.0.weight
    if any(k.startswith("features.") and ".block." in k for k in keys) or ("features.0.0.weight" in keys):
        return "efficientnet_b0"

    # ResNet
    if "conv1.weight" in keys or any(k.startswith("layer1.") for k in keys):
        return "resnet18"

    # Simple CNN
    if any(k.startswith("features.") for k in keys):
        return "deepfakecnn"

    return "unknown"


def normalize_state_dict_keys(state_dict: dict) -> dict:
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


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "backbone"):
            maybe = ckpt_obj.get(key)
            if isinstance(maybe, dict):
                return normalize_state_dict_keys(maybe)
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return normalize_state_dict_keys(ckpt_obj)
    raise RuntimeError(f"Checkpoint formatı desteklenmiyor: {type(ckpt_obj)}")


def infer_num_classes_from_state_dict(state_dict: dict) -> int:
    # ResNet
    w = state_dict.get("fc.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) > 0:
        return int(w.shape[0])

    # SimpleFaceCNN (son layer)
    w = state_dict.get("classifier.4.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) > 0:
        return int(w.shape[0])

    # EfficientNet-B0 (torchvision): classifier.1 = Linear(out=num_classes)
    w = state_dict.get("classifier.1.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) > 0:
        return int(w.shape[0])

    raise RuntimeError(
        "Checkpoint'ten num_classes çıkarılamadı (fc.weight / classifier.4.weight / classifier.1.weight yok)."
    )


def list_classes_from_data_root(data_root: str, split: str = "test"):
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        return None
    classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    classes = sorted(classes)
    return classes if classes else None


@st.cache_resource
def load_model(ckpt_path: str, device_str: str):
    device = torch.device(device_str)
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt_obj)

    arch = detect_arch(state_dict)
    num_classes = infer_num_classes_from_state_dict(state_dict)
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "deepfakecnn":
        m = SimpleFaceCNN(num_classes=num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        # torchvision EfficientNet classifier: Sequential(Dropout, Linear)
        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential) and len(m.classifier) >= 2:
            if isinstance(m.classifier[1], nn.Linear):
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            else:
                raise RuntimeError("EfficientNet classifier[1] Linear değil, beklenmeyen yapı.")
        else:
            raise RuntimeError("EfficientNet classifier yapısı beklenmeyen formatta.")
    else:
        raise RuntimeError(
            f"Checkpoint mimarisi anlaşılamadı. İlk birkaç key: {list(state_dict.keys())[:5]}"
        )

    load_res = m.load_state_dict(state_dict, strict=False)
    m.to(device)
    m.eval()
    return m, arch, num_classes, load_res
def sample_video_frames(video_bytes: bytes, frames_per_video: int = 20):
    if not HAS_CV2:
        raise RuntimeError("Video için opencv-python (cv2) gerekli.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Video açılamadı (cv2.VideoCapture).")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 0:
        idxs = np.linspace(0, total - 1, num=frames_per_video, dtype=int)
        idxs = np.unique(idxs)
    else:
        idxs = None

    frames = []
    if idxs is not None:
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    else:
        # frame sayısı okunamadıysa: ilk N frame
        while len(frames) < frames_per_video:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if not frames:
        raise RuntimeError("Videodan frame okunamadı.")
    return frames

def get_transform(arch: str, img_size: int):
    if arch in ("resnet18", "efficientnet_b0"):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    # simplecnn
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])



def preprocess_image(pil_img: Image.Image, *, tv_tfm):
    return tv_tfm(pil_img.convert("RGB"))


# ---- UI ----
st.set_page_config(page_title="Model Viewer", layout="centered")
st.title("Deepfake tespiti")

pth_files = sorted(glob.glob("*.pth"))
if not pth_files:
    st.warning("Bu klasörde .pth bulunamadı. app.py ile aynı klasöre checkpoint koy.")
    st.stop()

with st.sidebar:
    st.header("Ayarlar")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
                              index=0 if device_str == "cpu" else 1)

    img_size = st.number_input("IMG_SIZE", min_value=64, max_value=512, value=224, step=8)

    data_root = r"C:\ffpp_faces_splits_video"

    classes = list_classes_from_data_root(data_root, split="test")
    if not classes:
        st.caption("DATA_ROOT okunamadı; varsayılan ayarlar kullanılacak.")

    use_face = st.checkbox("Yüz kırp (MTCNN)", value=True, disabled=not HAS_MTCNN)
    if use_face and not HAS_MTCNN:
        st.warning("facenet_pytorch yok; yüz kırpma kapalı.")

ckpt = st.selectbox("Checkpoint seç (.pth)", pth_files)

input_type = st.radio("Girdi türü", ["Resim", "Video"], horizontal=True)

uploaded_img = None
uploaded_vid = None

if input_type == "Resim":
    uploaded_img = st.file_uploader("Resim yükle (jpg/png)", type=["jpg", "jpeg", "png"])
    if not uploaded_img:
        st.stop()
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Yüklenen resim", use_container_width=True)

else:
    if not HAS_CV2:
        st.error("Video için `opencv-python` lazım: `pip install opencv-python`")
        st.stop()

    frames_per_video = st.number_input("Videodan alınacak frame sayısı", min_value=1, max_value=200, value=20, step=1)
    uploaded_vid = st.file_uploader("Video yükle (mp4/avi/mov/mkv)", type=["mp4", "avi", "mov", "mkv"])
    if not uploaded_vid:
        st.stop()

    video_bytes = uploaded_vid.read()
    frames = sample_video_frames(video_bytes, frames_per_video=int(frames_per_video))

    img = frames[0].convert("RGB")  # örnek gösterim için
    st.image(img, caption="Örnek frame", use_container_width=True)

# Model
model, arch, ckpt_num_classes, load_res = load_model(ckpt, device_str=device_str)

# Decide class names (checkpoint ile uyumlu olacak şekilde)
if classes and len(classes) == ckpt_num_classes:
    class_names = classes
else:
    if classes and len(classes) != ckpt_num_classes:
        st.warning(
            f"DATA_ROOT class sayısı ({len(classes)}) ile checkpoint class sayısı ({ckpt_num_classes}) uyuşmuyor; "
            "checkpoint'e göre devam ediliyor."
        )
    if ckpt_num_classes == 2:
        class_names = ["original", "fake"]
    else:
        class_names = [f"class_{i}" for i in range(ckpt_num_classes)]

if getattr(load_res, "missing_keys", None) or getattr(load_res, "unexpected_keys", None):
    st.caption(f"load_state_dict missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}")

tv_tr = get_transform(arch, img_size)
device = torch.device(device_str)

# MTCNN (tek kez oluştur)
mtcnn = None
if use_face and HAS_MTCNN:
    mtcnn = MTCNN(keep_all=True, device=device, image_size=img_size, margin=20)

def maybe_crop(pil_img: Image.Image):
    if mtcnn is None:
        return pil_img
    boxes, _ = mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return pil_img
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box = boxes[int(np.argmax(areas))]
    x1, y1, x2, y2 = [int(v) for v in box]
    return pil_img.crop((x1, y1, x2, y2)).resize((img_size, img_size))
def find_real_index(class_names):
    lower = [c.lower() for c in class_names]
    for key in ("original", "real"):
        if key in lower:
            return lower.index(key)
    # binary modelde genelde 0 = real varsayımı
    if len(class_names) == 2:
        return 0
    return 0  # fallback (en azından app çalışsın)

def binary_pred_from_probs(probs, class_names):
    real_idx = find_real_index(class_names)          # "original" index'i
    pred_idx = int(np.argmax(probs))                 # 7 sınıftan en yüksek olan
    is_real = (pred_idx == real_idx)

    if is_real:
        label = "ORİJİNAL"
        conf = float(probs[real_idx])
    else:
        label = "DEEPFAKE"
        conf = float(probs[pred_idx])

    return label, conf, is_real




# Predict (resim/video)
if input_type == "Resim":
    face_img = maybe_crop(img)
    if use_face and HAS_MTCNN:
        st.image(face_img, caption="Kırpılan yüz", use_container_width=True)

    x0 = preprocess_image(face_img, tv_tfm=tv_tr)
    x = x0.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

else:
    probs_list = []
    cropped_faces = []
    frame_captions = []

    for i, fr in enumerate(frames):
        fr = fr.convert("RGB")
        face_fr = maybe_crop(fr)
        cropped_faces.append(face_fr)

        x0 = preprocess_image(face_fr, tv_tfm=tv_tr)
        x = x0.unsqueeze(0).to(device)
        with torch.no_grad():
            p = F.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

        probs_list.append(p)

        # Frame bazlı: original değilse DEEPFAKE
        frame_label, frame_conf, _ = binary_pred_from_probs(p, class_names)
        frame_pred_idx = int(np.argmax(p))
        frame_pred_name = class_names[frame_pred_idx] if frame_pred_idx < len(class_names) else str(frame_pred_idx)
        frame_captions.append(f"Frame {i+1} | {frame_label} ({frame_pred_name}) ")

    probs = np.mean(np.stack(probs_list, axis=0), axis=0)

    st.subheader("Videodaki kırpılmış yüzler")
    st.image(cropped_faces, caption=frame_captions, width=img_size)

label, conf, is_real = binary_pred_from_probs(probs, class_names)
is_df = not is_real

pred_idx = int(np.argmax(probs))

real_idx = find_real_index(class_names)


st.subheader("Sonuç")
st.write(f"Model: **{arch}**  |  Checkpoint: **{ckpt}**")
st.write("Sonuç:", label)
st.write("Deepfake mi?:", "EVET" if is_df else "HAYIR")

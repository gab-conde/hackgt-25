# full_pipeline_gdino_sam_da2.py
# Grounding DINO (zero-shot boxes) → SAM (pixel masks) → Depth-Anything V2 (relative depth)
# → plate-plane height (zero-centered) → optional rim→mm scale → robust volume/grams
# Saves: raw depth/height, previews, per-object CSV, mask PNGs, annotated JPEG.

import sys, os, io, time, csv, requests
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

DA2_REPO_DIR = "/Users/albertzheng/PyCharmMiscProject/Depth-Anything-V2"
if os.path.isdir(DA2_REPO_DIR) and DA2_REPO_DIR not in sys.path:
    sys.path.insert(0, DA2_REPO_DIR)
device = "cpu"
# -------- Grounding DINO --------
from groundingdino.util.inference import Model as GDINOModel
try:
    from supervision.detection.core import Detections  # newer API type
except Exception:
    Detections = None

# -------- SAM (Segment Anything) --------
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

from depth_anything_v2.dpt import DepthAnythingV2

# ===================== PATHS / ASSETS =====================
IMG_SRC = "/Users/albertzheng/Downloads/hot_dog.jpg"  # <-- your image

GDINO_CFG = "/Users/albertzheng/PyCharmMiscProject/weights/GroundingDINO_SwinT_OGC.cfg.py"
GDINO_WEIGHTS = "/Users/albertzheng/PyCharmMiscProject/weights/groundingdino_swint_ogc.pth"

# SAM
SAM_CKPT = "/Users/albertzheng/PyCharmMiscProject/weights/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
USE_SAM = True  # required in this pipeline

# Depth-Anything V2
DA2_MODEL_TYPE = "vits"  # vitl | vitb | vits (match your .pth)
DA2_CKPT = "/Users/albertzheng/PyCharmMiscProject/weights/depth_anything_v2_vits.pth"

# (Optional) plate rim height in mm for metric scaling (set to None to skip)
KNOWN_RIM_MM = 10   # e.g., 6.5


CFG_URLS = [
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.cfg.py",
]
WEIGHTS_URLS = [
    "https://huggingface.co/pengxian/grounding-dino/resolve/main/groundingdino_swint_ogc.pth",
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
]
SAM_URLS = [
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "https://huggingface.co/facebook/sam-vit-base/resolve/main/sam_vit_b_01ec64.pth",
]

# ===================== DETECTION PROMPTS =====================
CLASSES = [
    "paper plate",
    "hamburger bun",
    "hot dog bun",
]
BOX_THR  = 0.2
TEXT_THR = 0.2

# generic/ambiguous phrases we don’t want to keep
GENERIC = {"object", "thing", "unknown", "stuff"}


# ===================== GEOMETRY / PHYSICS =====================
PLATE_DIAM_CM = 21.6   # 8.5 in
H_MIN_M = 0.00
# We won't hard-clip globally; use a loose safety to avoid NaNs/explosions
GLOBAL_SAFETY_HMAX_M = 0.10  # 10 cm safety only
RHO_TABLE = {
    "cookie": 0.70, "biscuit": 0.65, "cracker": 0.55,
    "beef": 1.00, "meat": 1.00, "rice": 0.95, "kimchi": 0.60,
    "lettuce": 0.15, "banchan": 0.70
}


# ===================== DOWNLOADER =====================
def _download_one(url: str, dst: str, chunk=1024*1024) -> float:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[INFO] downloading {os.path.basename(dst)} from {url}")
    with requests.get(url, stream=True, timeout=180, headers={"User-Agent":"Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b: f.write(b)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"[INFO] saved to {dst} ({size_mb:.2f} MB)")
    return size_mb

def download_with_fallbacks(urls, dst, min_mb):
    last_err = None
    for url in urls:
        try:
            size_mb = _download_one(url, dst)
            if size_mb >= min_mb:
                return
            print(f"[WARN] File too small ({size_mb:.3f} MB) from {url}, trying next...")
        except Exception as e:
            print(f"[WARN] Download failed from {url}: {e}")
            last_err = e
    raise RuntimeError(f"All download sources failed for {dst}. Last error: {last_err}")

def ensure_gdino_assets():
    if not os.path.exists(GDINO_CFG) or os.path.getsize(GDINO_CFG) < 1024:
        download_with_fallbacks(CFG_URLS, GDINO_CFG, min_mb=0.001)
    else:
        print("[INFO] CFG present:", GDINO_CFG, f"({os.path.getsize(GDINO_CFG)} bytes)")
    if not os.path.exists(GDINO_WEIGHTS) or os.path.getsize(GDINO_WEIGHTS) < 10_000_000:
        download_with_fallbacks(WEIGHTS_URLS, GDINO_WEIGHTS, min_mb=10.0)
    else:
        print("[INFO] Weights present:", GDINO_WEIGHTS, f"({os.path.getsize(GDINO_WEIGHTS)/1e6:.1f} MB)")

def ensure_sam_assets():
    os.makedirs(os.path.dirname(SAM_CKPT), exist_ok=True)
    if not os.path.exists(SAM_CKPT) or os.path.getsize(SAM_CKPT) < 300_000_000:
        print("[INFO] SAM checkpoint missing or too small; downloading ViT-B...")
        download_with_fallbacks(SAM_URLS, SAM_CKPT, min_mb=300.0)
        print(f"[INFO] SAM checkpoint ready at {SAM_CKPT}")


# ===================== IO / DRAW =====================
def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        data = requests.get(path_or_url, timeout=15).content
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img

def draw_results(image_pil, dets, masks=None, plate_box=None):
    img = image_pil.copy()
    drw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    if masks is not None:
        overlay = np.array(img)
        for m in masks:
            if m is None: continue
            cnts, _ = cv2.findContours((m.astype(np.uint8))*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0,255,255), 2)
        img = Image.fromarray(overlay)
        drw = ImageDraw.Draw(img)

    for d in dets:
        x1,y1,x2,y2 = map(int, d["box"])
        is_plate = "plate" in d["label"].lower() or "dish" in d["label"].lower()
        color = "cyan" if is_plate else "lime"
        drw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        drw.text((x1+3, max(0,y1-18)), f"{d['label']} {d['score']:.2f}", fill=color, font=font)

    if plate_box:
        drw.rectangle(plate_box, outline="red", width=2)
        drw.text((plate_box[0], max(0, plate_box[1]-18)), "plate(scale)", fill="red", font=font)
    return img


# ===================== GROUNDING DINO WRAPPER =====================
class Detector:
    def __init__(self, cfg_path, weights_path):
        if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing GDINO assets.\n{cfg_path}\n{weights_path}")
        self.model = GDINOModel(
            model_config_path=cfg_path,
            model_checkpoint_path=weights_path,
            device="cpu"   # force CPU for compatibility
        )

    @torch.no_grad()
    def detect(self, bgr_img, classes, box_thr=0.40, text_thr=0.40):
        out = self.model.predict_with_classes(
            image=bgr_img,
            classes=classes,
            box_threshold=box_thr,
            text_threshold=text_thr
        )
        dets = []

        is_det_type = (Detections is not None and isinstance(out, Detections)) or (
            hasattr(out, "xyxy") and hasattr(out, "confidence")
        )
        if is_det_type:
            boxes  = out.xyxy
            scores = out.confidence
            N = len(boxes)

            labels = None
            if hasattr(out, "data") and isinstance(out.data, dict):
                for key in ("class_name", "labels", "phrases", "text", "phrase"):
                    val = out.data.get(key, None)
                    if val is not None and len(val) == N:
                        labels = [str(v) for v in val]; break
            if labels is None:
                labels = ["object"] * N
                class_ids = getattr(out, "class_id", None)
                if class_ids is not None and len(class_ids) == N:
                    for i, cid in enumerate(class_ids):
                        if cid is None: continue
                        try: labels[i] = str(classes[int(cid)])
                        except Exception: labels[i] = str(cid)

            for i in range(N):
                x1,y1,x2,y2 = boxes[i].tolist()
                dets.append({"box":[int(x1),int(y1),int(x2),int(y2)],
                             "score":float(scores[i]),
                             "label":labels[i]})
            return dets

        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                boxes, logits, phrases = out; scores = logits
            elif len(out) == 4:
                boxes, logits, phrases, scores = out
            else:
                raise RuntimeError(f"Unexpected GDINO output len={len(out)}")
            to_list = lambda x: x.detach().cpu().tolist() if hasattr(x, "detach") else x
            boxes = to_list(boxes); scores = to_list(scores)
            for i, b in enumerate(boxes):
                x1,y1,x2,y2 = map(int, b)
                lab = phrases[i] if isinstance(phrases, (list, tuple)) else str(phrases)
                s = float(scores[i]) if isinstance(scores, (list, tuple)) else float(scores)
                dets.append({"box":[x1,y1,x2,y2], "score":s, "label":lab})
            return dets

        if isinstance(out, dict):
            boxes  = out.get("boxes", [])
            phrases = out.get("labels", out.get("phrases", []))
            scores = out.get("scores", out.get("logits", []))
            to_list = lambda x: x.detach().cpu().tolist() if hasattr(x, "detach") else x
            boxes = to_list(boxes); scores = to_list(scores)
            for i, b in enumerate(boxes):
                x1,y1,x2,y2 = map(int, b)
                s = float(scores[i]) if isinstance(scores, (list, tuple)) else float(scores)
                lab = phrases[i] if isinstance(phrases, (list, tuple)) else str(phrases)
                dets.append({"box":[x1,y1,x2,y2], "score":s, "label":lab})
            return dets

        raise RuntimeError(f"Unexpected GroundingDINO output type: {type(out)}")


# ===================== GEOM UTILS / FILTERS =====================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw*ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms_xyxy(dets, iou_thresh=0.5):
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        # suppress ALL heavily-overlapping boxes, even if labels differ
        dets = [d for d in dets if iou_xyxy(best["box"], d["box"]) < iou_thresh]
    return keep


def bbox_to_circle(bbox):
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)//2; cy = (y1+y2)//2
    r = int(0.5 * min(x2-x1, y2-y1))
    return (cx, cy, r)

def get_plate_scale_and_circle_from_dets(image_pil, dets, plate_diam_cm=PLATE_DIAM_CM):
    for d in dets:
        if any(k in d["label"].lower() for k in ["plate","dish"]):
            cx, cy, r = bbox_to_circle(d["box"])  # use min side
            plate_px = 2 * r
            cm_per_px = plate_diam_cm / max(plate_px, 1)
            return d["box"], cm_per_px, (cx, cy, r)
    # (keep your HoughCircles fallback as-is)


    # Fallback: Hough circle on gray
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=100, param2=40, minRadius=80, maxRadius=2000)
    if circles is not None:
        x, y, r = np.round(circles[0,0]).astype("int")
        plate_px = 2 * r
        cm_per_px = plate_diam_cm / max(plate_px, 1)
        bbox = (x-r, y-r, x+r, y+r)
        print(f"[INFO] Plate via HoughCircles, cm/px = {cm_per_px:.4f}")
        return bbox, cm_per_px, (x, y, r)

    print("[WARN] Plate not detected. Using fallback scale 0.05 cm/px.")
    return None, 0.05, None


def is_plate_like(d, plate_box):
    lab = d["label"].lower()
    if "plate" in lab or "dish" in lab:
        return True
    if plate_box is None: return False
    return iou_xyxy(d["box"], plate_box) > 0.80

def coerce_cookie_label_if_small(d):
    x1,y1,x2,y2 = d["box"]; w,h = x2-x1, y2-y1
    aspect = w / max(1,h)
    if d["label"].lower() == "object" and 0.8 <= aspect <= 1.25 and max(w,h) < 450:
        d["label"] = "cookie"
    return d

def coerce_bun_label_by_shape(d, cm_per_px):
    """If label is generic/ambiguous, guess hot dog vs hamburger by shape/size."""
    lab = d["label"].lower()
    if lab in GENERIC or lab == "bun":
        x1, y1, x2, y2 = d["box"]
        w, h = x2 - x1, y2 - y1
        ar = max(w, h) / max(1, min(w, h))          # aspect ratio
        max_side_cm = max(w, h) * cm_per_px         # rough size

        if ar >= 1.5 and max_side_cm >= 8:          # long & skinny => hot dog bun
            d["label"] = "hot dog bun"
        elif ar <= 1.35 and 8 <= max_side_cm <= 16: # roundish burger-size => hamburger bun
            d["label"] = "hamburger bun"
    return d

def nms_one_label(dets, iou=0.5):
    """Standard NMS within one label (keeps multiple instances if they don't overlap much)."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou_xyxy(best["box"], d["box"]) < iou]
    return keep

def multi_label_nms(dets, iou_within=0.5, iou_cross=0.85):
    """
    Stage 1: NMS within each label (keeps multiple cookies/buns/etc).
    Stage 2: Deduplicate across labels (drop near-duplicate boxes that only differ by name).
    """
    # per-label NMS
    by_label = {}
    for d in dets:
        by_label.setdefault(d["label"].lower(), []).append(d)
    kept = []
    for lab, grp in by_label.items():
        kept.extend(nms_one_label(grp, iou=iou_within))

    # cross-label dedup
    kept = sorted(kept, key=lambda d: d["score"], reverse=True)
    final = []
    while kept:
        best = kept.pop(0)
        final.append(best)
        kept = [d for d in kept if iou_xyxy(best["box"], d["box"]) < iou_cross]
    return final


def local_object_height_mm(h_m, mask, plate_mask, cm_per_px,
                           ring_outer_cm=1.5, ring_inner_cm=0.5,
                           min_samples=500):
    """
    Subtract a local plate baseline measured in a thick ring (in *cm*) around
    the object. Returns a full-size height map (meters) where only the object's
    pixels are baseline-subtracted and clipped to >= 0.
    """
    m = np.asarray(mask, np.uint8)

    # convert desired ring thickness/offset in cm -> pixels
    rad_out = max(8, int(round(ring_outer_cm / cm_per_px)))   # e.g. 1.5 cm -> ~190 px for your image
    rad_in  = max(2, int(round(ring_inner_cm / cm_per_px)))   # keep an inner gap so ring excludes the cookie

    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad_out+1, 2*rad_out+1))
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad_in +1, 2*rad_in +1))

    outer = cv2.dilate(m, k_out, 1).astype(bool)
    inner = cv2.dilate(m, k_in,  1).astype(bool)
    ring  = outer & (~inner) & plate_mask

    ring_vals = h_m[ring]
    ring_vals = ring_vals[np.isfinite(ring_vals)]

    # if ring too small (e.g., cookie near rim edge), fall back to the clean-plate mask
    if ring_vals.size < min_samples:
        ring_vals = h_m[plate_mask]
        ring_vals = ring_vals[np.isfinite(ring_vals)]

    if ring_vals.size == 0:
        return h_m  # safest fallback

    base = float(np.median(ring_vals))  # meters
    h_local = h_m.copy()
    obj = m.astype(bool)
    h_local[obj] = np.clip(h_m[obj] - base, 0.0, None)
    return h_local


def label_to_rho(label: str) -> float:
    lab = label.lower()
    if lab in RHO_TABLE: return RHO_TABLE[lab]
    for k in ["cookie","biscuit","cracker","beef","meat","rice","kimchi","lettuce","banchan"]:
        if k in lab: return RHO_TABLE.get(k, 0.80)
    return 0.80

def estimate_grams(volume_cm3, label):
    rho = label_to_rho(label)
    return max(1.0, min(volume_cm3 * rho, 1500.0))


# ===================== SAM HELPERS =====================
def load_sam(model_type, ckpt_path, device="cpu"):
    if not SAM_AVAILABLE:
        raise RuntimeError("segment-anything not installed. pip install it or set USE_SAM=True.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")
    # normalize model type (tolerate 'vit_b_01ec64')
    mt = str(model_type).lower()
    if "vit_b" in mt:   mt = "vit_b"
    elif "vit_l" in mt: mt = "vit_l"
    elif "vit_h" in mt: mt = "vit_h"
    else:
        raise KeyError(f"Unknown SAM model type '{model_type}'. Use one of: vit_b, vit_l, vit_h.")
    sam = sam_model_registry[mt](checkpoint=ckpt_path)
    sam.to(device)
    return SamPredictor(sam)

def pad_box(shape_hw, box, pad_px=4):
    H, W = shape_hw
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(W - 1, x2 + pad_px)
    y2 = min(H - 1, y2 + pad_px)
    return [x1, y1, x2, y2]

def masks_from_sam(predictor: "SamPredictor", bgr_img, dets,
                   pad_px=6, close_px=3, erode_px=1, min_area=64):
    """
    Padded boxes → SAM → closing → tiny erosion → remove speckles.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    H, W = bgr_img.shape[:2]
    masks = []
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, close_px), max(1, close_px))) if close_px > 0 else None
    k_erode = np.ones((max(1, erode_px), max(1, erode_px)), np.uint8) if erode_px > 0 else None

    for d in dets:
        if any(k in d["label"].lower() for k in ["plate","dish"]):
            masks.append(None)
            continue
        x1, y1, x2, y2 = pad_box((H, W), d["box"], pad_px=pad_px)
        sam_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        m, _, _ = predictor.predict(box=sam_box, point_coords=None, point_labels=None, multimask_output=False)
        m = (m[0] > 0).astype(np.uint8)

        if k_close is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)
        if k_erode is not None:
            m = cv2.erode(m, k_erode, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        masks.append(keep.astype(bool))
    return masks


# ===================== DEPTH ANYTHING V2 LOADER / INFER =====================
# NOTE: We rely on the repo's own infer_image() so we don't import extra transforms.
# Make sure DA2_REPO_DIR points to the cloned repo that contains the 'depth_anything_v2' folder.

_DA2_MODEL = None

@torch.no_grad()
def get_da2_model(encoder: str, ckpt_path: str, device: str = "cpu"):
    """
    Build or return a cached Depth-Anything V2 model.
    encoder: 'vits' | 'vitb' | 'vitl' (must match your checkpoint filename)
    """
    global _DA2_MODEL
    if _DA2_MODEL is not None:
        return _DA2_MODEL

    from depth_anything_v2.dpt import DepthAnythingV2  # import from the repo on sys.path

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    if encoder not in model_configs:
        raise ValueError(f"DA2 encoder '{encoder}' not in {list(model_configs.keys())}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DA2 checkpoint missing: {ckpt_path}")

    print(f"[INFO] Loading Depth-Anything V2 ({encoder}) from {ckpt_path}")
    m = DepthAnythingV2(**model_configs[encoder])
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m = m.to(device).eval()
    m.device = torch.device(device)  # <-- important for infer_image() internals
    _DA2_MODEL = m
    return _DA2_MODEL


@torch.no_grad()
def da2_infer_depth(image_pil: "Image.Image", model_type=DA2_MODEL_TYPE, ckpt=DA2_CKPT, device="cpu"):
    """
    Returns HxW float32 depth (relative; larger => farther), aligned to the input image.
    Uses DA-V2's built-in infer_image(), per the official README.
    """
    model = get_da2_model(model_type, ckpt, device=device)
    # DA-V2 example code feeds a BGR image (cv2.imread). We'll match that.
    bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    depth = model.infer_image(bgr)  # HxW raw depth map (numpy)
    return depth.astype("float32")



# ===================== HEIGHT MAP / ZERO-CENTER / RIM CAL =====================
def plate_core_mask(shape, plate_circle, frac=0.55):
    H, W = shape
    cx, cy, r = plate_circle
    yy, xx = np.mgrid[0:H, 0:W]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    return dist2 <= (r*frac)**2

def plate_rim_band(shape, plate_circle, r_inner=0.80, r_outer=0.98):
    H, W = shape
    cx, cy, r = plate_circle
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx)**2 + (yy - cy)**2
    return (d2 >= (r * r_inner)**2) & (d2 <= (r * r_outer)**2)

def calibrate_scale_from_plate_rim_crest(
    h_zero, plate_circle, clean_plate_mask, food_masks, known_rim_mm,
    r_inner=0.88, r_outer=0.97, use_top_frac=0.02
):
    """
    Use only the *crest* of the rim:
    - tight annulus near the crest (88–97% of radius)
    - take the top X% of heights in that annulus (removes sloped pixels)
    - exclude food everywhere
    Returns (mm_per_unit, possibly flipped h_zero).
    """
    if known_rim_mm is None:
        raise RuntimeError("KNOWN_RIM_MM must be set (mm).")

    H, W = h_zero.shape
    cx, cy, r = plate_circle
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx)**2 + (yy - cy)**2
    rim = (d2 >= (r * r_inner)**2) & (d2 <= (r * r_outer)**2)

    # exclude food
    if food_masks:
        k = np.ones((4, 4), np.uint8)
        u = np.zeros((H, W), np.uint8)
        for m in food_masks:
            if m is not None:
                u |= cv2.dilate(m.astype(np.uint8), k, 1)
        rim &= (u == 0)

    base_vals = h_zero[clean_plate_mask]
    rim_vals  = h_zero[rim]
    base_vals = base_vals[np.isfinite(base_vals)]
    rim_vals  = rim_vals[np.isfinite(rim_vals)]
    if base_vals.size < 500 or rim_vals.size < 500:
        raise RuntimeError("Not enough clean pixels for rim calibration.")

    base_med = float(np.median(base_vals))     # ~0
    rel = rim_vals - base_med

    # keep only the top X% (crest)
    rel_sorted = np.sort(rel)
    k = max(1, int(len(rel_sorted) * (1.0 - use_top_frac)))
    crest = rel_sorted[k:]
    if crest.size == 0:
        raise RuntimeError("Rim crest selection empty; relax r_inner/r_outer or use_top_frac.")

    ref = float(np.median(crest))
    print(f"[CAL] base_med={base_med:.6f} crest_med={ref:.6f}  known_rim_mm={known_rim_mm:.2f}")

    if not np.isfinite(ref) or abs(ref) < 1e-6:
        raise RuntimeError("Rim signal too small.")
    if ref < 0:
        h_zero = -h_zero
        ref = -ref
        print("[CAL] NOTE: inverted sign (rim appeared below base).")

    s_mm_per_unit = float(known_rim_mm) / ref
    return max(0.01, min(500.0, s_mm_per_unit)), h_zero





def height_map_from_plate(depth_map, plate_circle, cm_per_px, food_masks=None):
    """
    Fit plate plane, compute zero-centered height above the plate (unclipped),
    and return masks so calibration can exclude food properly.
    """
    H, W = depth_map.shape
    cx, cy, r = plate_circle
    depth_sm = cv2.medianBlur(depth_map.astype(np.float32), 3)

    yy, xx = np.mgrid[0:H, 0:W]
    plate_mask = (xx - cx)**2 + (yy - cy)**2 <= int((r * 0.90)**2)

    # Union of food masks (dilated) to exclude from the plate base & rim
    clean_plate_mask = plate_mask.copy()
    if food_masks:
        k = np.ones((6, 6), np.uint8)
        union = np.zeros_like(clean_plate_mask, dtype=np.uint8)
        for m in food_masks:
            if m is None: continue
            union |= cv2.dilate(np.logical_and(m, plate_mask).astype(np.uint8), k, iterations=1)
        clean_plate_mask = np.logical_and(plate_mask, union == 0)

    ys, xs = np.nonzero(plate_mask)
    if xs.size < 500:
        # Fallback: global median plane
        z_bg = np.median(depth_sm[np.isfinite(depth_sm)])
        h_raw = z_bg - depth_sm
        base = float(np.median(h_raw[clean_plate_mask])) if np.any(clean_plate_mask) else float(np.median(h_raw[plate_mask]))
        h_zero = h_raw - base
        return h_zero, plate_mask, clean_plate_mask

    # Plane fit in metricized x,y (only matters for good conditioning)
    xs_m = (xs * cm_per_px) / 100.0
    ys_m = (ys * cm_per_px) / 100.0
    Z = depth_sm[ys, xs]
    A = np.c_[xs_m, ys_m, np.ones_like(xs_m)]
    a, b, c = np.linalg.lstsq(A, Z, rcond=None)[0]
    X_m = (xx * cm_per_px) / 100.0
    Y_m = (yy * cm_per_px) / 100.0
    Z_plane = a * X_m + b * Y_m + c

    h_raw = Z_plane - depth_sm  # relative units
    base = float(np.median(h_raw[clean_plate_mask])) if np.any(clean_plate_mask) else float(np.median(h_raw[plate_mask]))
    h_zero = h_raw - base  # zero at clean plate
    return h_zero, plate_mask, clean_plate_mask



# ===================== ROBUST VOLUME INTEGRATION =====================
def winsorize_heights(h_vals, p_hi=97.0):
    if h_vals.size == 0: return h_vals
    hi = np.percentile(h_vals, p_hi)
    return np.minimum(h_vals, hi)

def mad_mm(h_vals):
    if h_vals.size == 0: return 0.0
    med = np.median(h_vals)
    return 1.4826 * np.median(np.abs(h_vals - med)) * 1000.0

def huber_weights(h_vals_mm, delta_mm):
    if delta_mm <= 1e-6:
        return np.ones_like(h_vals_mm)
    r = h_vals_mm / delta_mm
    w = np.ones_like(r)
    mask = r > 1.0
    w[mask] = 1.0 / r[mask]
    return w

def integrate_volume_cm3_over_mask_robust(h_m, mask, cm_per_px, p_hi=97.0, huber_k=2.5):
    """
    Robust volume = (weighted mean height) * area.
    We normalize by sum of weights so magnitude isn't arbitrarily shrunk.
    """
    h = h_m[mask]
    if h.size == 0:
        return 0.0

    # Trim extreme highs only
    hi = np.percentile(h, p_hi)
    h = np.minimum(h, hi)

    # Huber weights around the object's own median (in mm)
    h_mm = h * 1000.0
    med = np.median(h_mm)
    mad = 1.4826 * np.median(np.abs(h_mm - med)) + 1e-6
    delta = max(huber_k * mad, 1.0)

    r = (h_mm - med) / delta
    w = np.ones_like(r)
    big = np.abs(r) > 1.0
    w[big] = 1.0 / np.abs(r[big])

    # --- normalized weighted mean height (in meters) ---
    wsum = float(w.sum())
    if wsum < 1e-9:
        h_mean_m = float(h.mean())
    else:
        h_mean_m = float((h * w).sum() / wsum)

    area_cm2 = float(mask.sum()) * (cm_per_px ** 2)
    vol_cm3 = h_mean_m * 100.0 * area_cm2  # m→cm (×100) * area in cm^2
    return max(0.0, vol_cm3)


# ===================== EXPORT HELPERS =====================
def save_depth_visuals(depth_map, h_m, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    np.save(out_prefix + "_depth.npy", depth_map)
    np.save(out_prefix + "_height.npy", h_m)

    d = depth_map.copy()
    valid = np.isfinite(d)
    lo, hi = np.percentile(d[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    d = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
    d_img = (d * 255).astype(np.uint8)
    d_vis = cv2.applyColorMap(d_img, cv2.COLORMAP_JET)
    cv2.imwrite(out_prefix + "_depth.png", d_vis)

    h = h_m.copy()
    h[h < 0] = 0
    hmax = float(np.nanmax(h)) if np.isfinite(h).any() else 1.0
    hm = np.clip(h / max(1e-6, hmax), 0, 1)
    h_img = (hm * 255).astype(np.uint8)
    h_vis = cv2.applyColorMap(h_img, cv2.COLORMAP_JET)
    cv2.imwrite(out_prefix + "_height.png", h_vis)

def mask_to_png(mask, path):
    m = (mask.astype(np.uint8) * 255)
    cv2.imwrite(path, m)

def object_depth_stats(depth_map, h_m, mask, p_hi=97.0, huber_k=2.5):
    z = depth_map[mask]
    h = h_m[mask]
    if h.size == 0:
        return {"pixels": 0, "depth_mean": 0.0, "depth_median": 0.0, "depth_std": 0.0,
                "height_mean_mm": 0.0, "height_median_mm": 0.0,
                "height_p95_mm": 0.0, "height_max_mm": 0.0}

    # same trimming
    hi = np.percentile(h, p_hi)
    h = np.minimum(h, hi)

    # same weights
    h_mm = h * 1000.0
    med = np.median(h_mm)
    mad = 1.4826 * np.median(np.abs(h_mm - med)) + 1e-6
    delta = max(huber_k * mad, 1.0)
    r = (h_mm - med) / delta
    w = np.ones_like(r); w[np.abs(r) > 1.0] = 1.0 / np.abs(r[np.abs(r) > 1.0])
    wsum = float(w.sum())

    h_mean_mm = float((h_mm * w).sum() / wsum) if wsum > 1e-9 else float(h_mm.mean())
    h_median_mm = float(np.median(h_mm))  # robust center

    return {
        "pixels": int(mask.sum()),
        "depth_mean": float(np.mean(z)),
        "depth_median": float(np.median(z)),
        "depth_std": float(np.std(z)),
        "height_mean_mm": h_mean_mm,
        "height_median_mm": h_median_mm,
        "height_p95_mm": float(np.percentile(h_mm, 95)),
        "height_max_mm": float(np.max(h_mm)),
    }


def summarize_results(results):
    """
    Return compact list of {name, volume_cm3, grams} (skip plates).
    """
    out = []
    for r in results:
        name = r["label"]
        if "plate" in name.lower() or "dish" in name.lower():
            continue
        out.append({
            "name": name,
            "volume_cm3": float(round(r["volume_cm3"], 2)),
            "grams": float(round(r["grams"], 2)),
        })
    return out


def _union_mask(masks, shape, dilate=3):
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    k = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    u = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        if m is None:
            continue
        mm = m.astype(np.uint8)
        if k is not None:
            mm = cv2.dilate(mm, k, iterations=1)
        u |= mm
    return u

def auto_fix_sign(h_zero, plate_circle, clean_plate_mask, food_masks):
    """
    If rim appears below base, or food median is negative, flip signs so food > 0.
    Returns (h_zero_pos, flipped:boolean, diagnostics: dict).
    """
    H, W = h_zero.shape
    rim = plate_rim_band(h_zero.shape, plate_circle, r_inner=0.85, r_outer=0.98)

    # exclude food from rim too
    food_u = _union_mask(food_masks, (H, W), dilate=4).astype(bool)
    rim &= ~food_u

    base_vals = h_zero[clean_plate_mask]
    rim_vals  = h_zero[rim]
    food_vals = h_zero[food_u]

    base_med = np.median(base_vals[np.isfinite(base_vals)]) if base_vals.size else 0.0
    rim_p90  = np.percentile(rim_vals[np.isfinite(rim_vals)], 90) if rim_vals.size else 0.0
    food_med = np.median(food_vals[np.isfinite(food_vals)]) if food_vals.size else 0.0

    # Expect rim above base (positive) and food above base (positive).
    flip = (rim_p90 - base_med) < 0 or (food_med - base_med) < 0
    if flip:
        h_zero = -h_zero

    diag = dict(base_med=float(base_med), rim_p90=float(rim_p90), food_med=float(food_med))
    return h_zero, flip, diag

def local_object_height_m(h_m, mask, plate_mask, cm_per_px,
                          ring_outer_cm=1.5, ring_inner_cm=0.5,
                          min_samples=200, safety_max_m=0.10):
    """
    For one object, estimate its height above the nearby plate surface
    by subtracting the median height in a ring around the object.
    Returns a *full-frame* map (meters) with this local baseline removed.
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = h_m.shape

    # distance (in px) from the object
    inv = (~mask).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    ring_out_px = int(round(ring_outer_cm / cm_per_px))
    ring_in_px  = int(round(ring_inner_cm  / cm_per_px))
    ring = (dist > ring_in_px) & (dist <= ring_out_px)

    # restrict to plate pixels only
    ring &= plate_mask

    ring_vals = h_m[ring]
    if ring_vals.size < min_samples:
        # widen ring once if too few samples
        ring_out_px = int(round(max(ring_out_px, ring_in_px + 10) * 1.5))
        ring = (dist > ring_in_px) & (dist <= ring_out_px) & plate_mask
        ring_vals = h_m[ring]

    # fallback baseline
    baseline = float(np.median(ring_vals)) if ring_vals.size else 0.0

    # DEBUG: uncomment to inspect the baseline you’re subtracting
    # print(f"[DBG] ring samples={ring_vals.size}, baseline_med_mm={baseline*1000:.2f}")

    h_local = h_m - baseline
    h_local = np.clip(h_local, 0.0, safety_max_m)  # keep positive, cap to safety
    return h_local


# ===================== MAIN =====================
def main():
    ensure_gdino_assets()
    ensure_sam_assets()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Load image
    image = load_image(IMG_SRC)
    bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 1) Detections
    # 1) Detections
    gdino = Detector(GDINO_CFG, GDINO_WEIGHTS)
    dets = gdino.detect(bgr, classes=CLASSES, box_thr=BOX_THR, text_thr=TEXT_THR)
    dets = [coerce_cookie_label_if_small(d) for d in dets]
    print(f"[INFO] {len(dets)} raw detections")
    for d in dets:
        print(f"  - {d['label']}  {d['score']:.2f}  {d['box']}")

    # 2) Plate scale + circle (we need cm/px before shape-based coercion)
    plate_box, cm_per_px, plate_circle = get_plate_scale_and_circle_from_dets(image, dets, PLATE_DIAM_CM)
    if plate_circle is None and plate_box is not None:
        plate_circle = bbox_to_circle(plate_box)
    plate_box_for_draw = plate_box

    # 2b) Coerce generic "object" boxes into bun types *using geometry*
    dets = [coerce_bun_label_by_shape(d, cm_per_px) for d in dets]

    # 2c) Now drop the truly generic leftovers
    dets = [d for d in dets if d["label"].lower() not in GENERIC]

    # 3) Remove plate-like non-plate
    if plate_box is not None:
        dets = [d for d in dets if not (is_plate_like(d, plate_box) and "plate" not in d["label"].lower())]

    # 4) Two-stage NMS: within-label then cross-label
    dets = multi_label_nms(dets, iou_within=0.5, iou_cross=0.85)

    print(f"[INFO] {len(dets)} detections after two-stage NMS")
    for d in dets:
        print(f"  - {d['label']}  {d['score']:.2f}  {d['box']}")

    # 5) SAM masks (REQUIRED)
    if not USE_SAM:
        raise RuntimeError("This pipeline requires SAM masks. Set USE_SAM=True and provide a valid checkpoint.")
    predictor = load_sam(SAM_MODEL_TYPE, SAM_CKPT, device=device)
    masks = masks_from_sam(predictor, bgr, dets, pad_px=6, close_px=3, erode_px=1, min_area=64)

    # Build a list of food masks (exclude plate/dish) for zero-centering & rim calibration
    food_masks_for_zero = []
    for d, m in zip(dets, masks):
        if m is None:
            continue
        lab = d["label"].lower()
        if "plate" in lab or "dish" in lab:
            continue
        food_masks_for_zero.append(m)

    # 6) Depth-Anything V2 relative depth
    t0 = time.time()
    depth_rel = da2_infer_depth(image, model_type=DA2_MODEL_TYPE, ckpt=DA2_CKPT, device=device)
    print(f"[INFO] Depth-Anything inference took {(time.time()-t0)*1000:.1f} ms")

    # 7) Height map from plate plane (zero-centered, unscaled)
    h_zero, plate_mask, clean_plate_mask = height_map_from_plate(
        depth_rel, plate_circle, cm_per_px, food_masks=food_masks_for_zero
    )

    # --- auto-fix sign BEFORE any clipping ---
    h_zero, flipped, diag = auto_fix_sign(h_zero, plate_circle, clean_plate_mask, food_masks_for_zero)
    print(f"[SIGN] base_med={diag['base_med']:.6f} rim_p90={diag['rim_p90']:.6f} "
          f"food_med={diag['food_med']:.6f}  flipped={'YES' if flipped else 'no'}")

    # 7b) REQUIRED: scale to metric via plate rim height (mm). If it fails, fallback to 1.0.
    try:
        s_mm_per_unit, h_zero = calibrate_scale_from_plate_rim_crest(
            h_zero, plate_circle, clean_plate_mask, food_masks_for_zero,
            KNOWN_RIM_MM, r_inner=0.88, r_outer=0.97, use_top_frac=0.02
        )
        print(f"[INFO] Plate-rim calibration scale = {s_mm_per_unit:.3f} mm/unit")

        print(f"[INFO] Plate-rim calibration scale = {s_mm_per_unit:.3f} mm/unit")
    except Exception as e:
        print(f"[WARN] Rim calibration failed: {e}. Using fallback scale 1.0 mm/unit.")
        s_mm_per_unit = 1.0

    # 7c) Convert to meters and lightly clip (pos only)
    h_m = (h_zero * s_mm_per_unit) / 1000.0  # meters
    print(f"[DBG] h_m pre-clip min/max: {np.nanmin(h_m):.6f}/{np.nanmax(h_m):.6f}")
    h_m = np.clip(h_m, 0.0, GLOBAL_SAFETY_HMAX_M)
    print(f"[DBG] h_m post-clip min/max: {np.nanmin(h_m):.6f}/{np.nanmax(h_m):.6f}")

    # 8) Save raw depth + height previews
    out_prefix = os.path.splitext(IMG_SRC)[0]
    save_depth_visuals(depth_rel, h_m, out_prefix)
    print(f"[INFO] Depth/height saved as {out_prefix}_depth.npy/png and {out_prefix}_height.npy/png")

    # 9) Integrate per detection; export CSV + mask PNGs
    # 9) Integrate per detection; export CSV + mask PNGs
    rows = []
    results = []

    # constants for local ring (in cm)
    ring_out_cm = 1.5
    ring_in_cm = 0.5
    min_ring_samples = 200

    # precompute a wide plate mask (≤ 0.98 * R) so the ring can see the crest
    H, W = h_m.shape
    cx, cy, r = plate_circle
    yy, xx = np.mgrid[0:H, 0:W]
    plate_mask_wide = (xx - cx) ** 2 + (yy - cy) ** 2 <= int((r * 0.98) ** 2)

    for i, (d, mask) in enumerate(zip(dets, masks)):
        lab = d["label"].lower()
        if "plate" in lab or "dish" in lab:
            continue

        # --- guard mask ---
        if mask is None:
            print(f"[WARN] mask is None for {d['label']} (idx={i}); skipping")
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or mask.sum() == 0:
            print(f"[WARN] empty mask for {d['label']} (idx={i}); skipping")
            continue

        # distance transform from object (so ring excludes object)
        inv = (~mask).astype(np.uint8) * 255
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

        # ring in pixels
        ring_out_px = int(round(ring_out_cm / cm_per_px))
        ring_in_px = int(round(ring_in_cm / cm_per_px))
        ring = (dist > ring_in_px) & (dist <= ring_out_px)
        # restrict to wide plate mask so we can include the crest neighborhood
        ring &= plate_mask_wide

        ring_vals = h_m[ring]
        if ring_vals.size < min_ring_samples:
            # widen once if too few samples
            ring_out_px_wide = int(round(max(ring_out_px, ring_in_px + 10) * 1.5))
            ring = (dist > ring_in_px) & (dist <= ring_out_px_wide) & plate_mask_wide
            ring_vals = h_m[ring]

        baseline = float(np.median(ring_vals)) if ring_vals.size else 0.0
        # --- DEBUG: print baseline minus cookie med, so you can see subtraction working
        cookie_med = float(np.median(h_m[mask])) if mask.sum() else 0.0
        print(f"[DBG][{i}] {d['label']}: cm/px={cm_per_px:.4f}, ring_in_px={ring_in_px}, "
              f"ring_out_px={ring_out_px}, ring_samples={ring_vals.size}, "
              f"baseline_med={baseline * 1000:.1f} mm, cookie_med_global={cookie_med * 1000:.1f} mm")

        # --- per-object local baseline (meters): subtract baseline everywhere, then clip
        h_local = h_m - baseline
        h_local = np.clip(h_local, 0.0, GLOBAL_SAFETY_HMAX_M)

        # --- stats from LOCAL heights ---
        stats = object_depth_stats(depth_rel, h_local, mask)

        # --- robust volume from LOCAL heights ---
        vol = integrate_volume_cm3_over_mask_robust(
            h_local, mask, cm_per_px, p_hi=97.0, huber_k=2.5
        )
        if not np.isfinite(vol):
            print(f"[WARN] non-finite volume for {d['label']} (idx={i}); skipping")
            continue
        g = estimate_grams(vol, d["label"])

        area_cm2 = float(mask.sum()) * (cm_per_px ** 2)
        avg_h_mm = float((vol / area_cm2) * 10.0) if area_cm2 > 1e-9 else 0.0

        # save mask preview
        mask_path = f"{out_prefix}_mask_{i}.png"
        mask_to_png(mask, mask_path)

        results.append({
            "label": d["label"],
            "score": float(d["score"]),
            "box": [int(v) for v in d["box"]],
            "volume_cm3": float(vol),
            "grams": float(g),
        })

        # CSV row
        rows.append({
            "label": d["label"], "score": float(d["score"]),
            "x1": int(d["box"][0]), "y1": int(d["box"][1]),
            "x2": int(d["box"][2]), "y2": int(d["box"][3]),
            "area_cm2": float(area_cm2),
            "avg_height_mm": float(avg_h_mm),
            "volume_cm3": float(vol), "grams": float(g),
            "pixels": int(stats.get("pixels", 0)),
            "depth_mean": float(stats.get("depth_mean", 0.0)),
            "depth_median": float(stats.get("depth_median", 0.0)),
            "depth_std": float(stats.get("depth_std", 0.0)),
            "height_mean_mm": float(stats.get("height_mean_mm", 0.0)),
            "height_median_mm": float(stats.get("height_median_mm", 0.0)),
            "height_p95_mm": float(stats.get("height_p95_mm", 0.0)),
            "height_max_mm": float(stats.get("height_max_mm", 0.0)),
            "mask_png": mask_path,
        })

    # nice console summary
    results.sort(key=lambda r: r["grams"], reverse=True)
    for r in results:
        print(f"[RESULT] {r['label']:<28} vol≈{r['volume_cm3']:7.1f} cm^3   "
              f"wt≈{r['grams']:6.2f} g   box={r['box']}")

    # write CSV
    if rows:
        csv_path = out_prefix + "_objects5.csv"
        fieldnames = [
            "label", "score", "x1", "y1", "x2", "y2",
            "area_cm2", "avg_height_mm", "volume_cm3", "grams",
            "pixels", "depth_mean", "depth_median", "depth_std",
            "height_mean_mm", "height_median_mm", "height_p95_mm", "height_max_mm",
            "mask_png"
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[INFO] Object stats saved to {csv_path}")
    else:
        print("[WARN] No rows to write; all masks skipped or invalid.")

    # 10) Save visualization (boxes + mask contours + plate box)
    vis = draw_results(image, dets, masks=masks, plate_box=plate_box_for_draw)
    out_path = out_prefix + "_gdino_sam_annotated5.jpg"
    vis.save(out_path, quality=95)
    print(f"[INFO] Saved annotated image to {out_path}")

# ---------- CAMERA/STREAM INTEGRATION HELPERS ----------

def init_pipeline(device: str = None):
    """
    Load all heavy models once and return a handle.
    """
    ensure_gdino_assets()
    ensure_sam_assets()

    dev = device or ("cuda" if torch.cuda.is_available()
                     else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"[INIT] device: {dev}")

    # GroundingDINO
    gdino = Detector(GDINO_CFG, GDINO_WEIGHTS)

    # SAM
    predictor = load_sam(SAM_MODEL_TYPE, SAM_CKPT, device=dev)

    # Depth Anything V2 (cache inside get_da2_model)
    get_da2_model(DA2_MODEL_TYPE, DA2_CKPT, device=dev)

    return {"device": dev, "gdino": gdino, "sam": predictor}


@torch.no_grad()
def analyze_pil(image: Image.Image, models: dict, save_prefix: str = None):
    """
    Run your full pipeline on a PIL image using preloaded models.
    Returns (results, rows) like your main() builds.
    """
    device = models["device"]
    gdino  = models["gdino"]
    predictor = models["sam"]

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 1) Detections
    dets  = gdino.detect(bgr, classes=CLASSES, box_thr=BOX_THR, text_thr=TEXT_THR)
    dets  = [coerce_cookie_label_if_small(d) for d in dets]

    # 2) Scale from plate
    plate_box, cm_per_px, plate_circle = get_plate_scale_and_circle_from_dets(image, dets, PLATE_DIAM_CM)
    if plate_circle is None and plate_box is not None:
        plate_circle = bbox_to_circle(plate_box)
    plate_box_for_draw = plate_box

    # Coerce generic bun labels AFTER we know cm/px
    dets = [coerce_bun_label_by_shape(d, cm_per_px) for d in dets]
    dets = [d for d in dets if d["label"].lower() not in GENERIC]

    # 3) Remove plate-like non-plate
    if plate_box is not None:
        dets = [d for d in dets if not (is_plate_like(d, plate_box) and "plate" not in d["label"].lower())]

    # 4) Two-stage NMS (within- and cross-label)
    dets = multi_label_nms(dets, iou_within=0.5, iou_cross=0.85)

    # 5) SAM masks
    masks = masks_from_sam(predictor, bgr, dets, pad_px=6, close_px=3, erode_px=1, min_area=64)

    # Build food masks list
    food_masks_for_zero = []
    for d, m in zip(dets, masks):
        if m is None: continue
        lab = d["label"].lower()
        if "plate" in lab or "dish" in lab: continue
        food_masks_for_zero.append(m)

    # 6) Depth Anything V2
    depth_rel = da2_infer_depth(image, model_type=DA2_MODEL_TYPE, ckpt=DA2_CKPT, device=device)

    # 7) Height map zero-center, sign fix, rim scale
    h_zero, plate_mask, clean_plate_mask = height_map_from_plate(
        depth_rel, plate_circle, cm_per_px, food_masks=food_masks_for_zero
    )
    h_zero, flipped, diag = auto_fix_sign(h_zero, plate_circle, clean_plate_mask, food_masks_for_zero)

    try:
        s_mm_per_unit, h_zero = calibrate_scale_from_plate_rim_crest(
            h_zero, plate_circle, clean_plate_mask, food_masks_for_zero,
            KNOWN_RIM_MM, r_inner=0.88, r_outer=0.97, use_top_frac=0.02
        )
    except Exception as e:
        print(f"[WARN] Rim calibration failed: {e}. Using scale=1.0 mm/unit.")
        s_mm_per_unit = 1.0

    h_m = np.clip((h_zero * s_mm_per_unit) / 1000.0, 0.0, GLOBAL_SAFETY_HMAX_M)

    # 8) Optionally save previews
    if save_prefix:
        save_depth_visuals(depth_rel, h_m, save_prefix)
        vis = draw_results(image, dets, masks=masks, plate_box=plate_box_for_draw)
        vis.save(save_prefix + "_gdino_sam_annotated_cam.jpg", quality=95)

    # 9) Per-object integration
    rows, results = [], []

    # precompute a wide plate mask for local baseline
    H, W = h_m.shape
    cx, cy, r = plate_circle
    yy, xx = np.mgrid[0:H, 0:W]
    plate_mask_wide = (xx - cx)**2 + (yy - cy)**2 <= int((r * 0.98) ** 2)

    ring_out_cm, ring_in_cm = 1.5, 0.5
    min_ring_samples = 200

    for i, (d, mask) in enumerate(zip(dets, masks)):
        lab = d["label"].lower()
        if "plate" in lab or "dish" in lab or mask is None or mask.sum() == 0:
            continue

        inv = (~mask).astype(np.uint8) * 255
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

        ring_out_px = int(round(ring_out_cm / cm_per_px))
        ring_in_px  = int(round(ring_in_cm  / cm_per_px))
        ring = (dist > ring_in_px) & (dist <= ring_out_px) & plate_mask_wide
        ring_vals = h_m[ring]
        if ring_vals.size < min_ring_samples:
            ring_out_px = int(round(max(ring_out_px, ring_in_px + 10) * 1.5))
            ring = (dist > ring_in_px) & (dist <= ring_out_px) & plate_mask_wide
            ring_vals = h_m[ring]
        baseline = float(np.median(ring_vals)) if ring_vals.size else 0.0

        h_local = np.clip(h_m - baseline, 0.0, GLOBAL_SAFETY_HMAX_M)

        stats = object_depth_stats(depth_rel, h_local, mask)
        vol = integrate_volume_cm3_over_mask_robust(h_local, mask, cm_per_px, p_hi=97.0, huber_k=2.5)
        g = estimate_grams(vol, d["label"])
        area_cm2 = float(mask.sum()) * (cm_per_px ** 2)
        avg_h_mm = float((vol / max(area_cm2, 1e-9)) * 10.0)

        results.append({"label": d["label"], "score": float(d["score"]),
                        "box": [int(v) for v in d["box"]],
                        "volume_cm3": float(vol), "grams": float(g)})

        rows.append({
            "label": d["label"], "score": float(d["score"]),
            "x1": int(d["box"][0]), "y1": int(d["box"][1]),
            "x2": int(d["box"][2]), "y2": int(d["box"][3]),
            "area_cm2": float(area_cm2), "avg_height_mm": float(avg_h_mm),
            "volume_cm3": float(vol), "grams": float(g),
            "pixels": int(stats.get("pixels", 0)),
            "depth_mean": float(stats.get("depth_mean", 0.0)),
            "depth_median": float(stats.get("depth_median", 0.0)),
            "depth_std": float(stats.get("depth_std", 0.0)),
            "height_mean_mm": float(stats.get("height_mean_mm", 0.0)),
            "height_median_mm": float(stats.get("height_median_mm", 0.0)),
            "height_p95_mm": float(stats.get("height_p95_mm", 0.0)),
            "height_max_mm": float(stats.get("height_max_mm", 0.0)),
        })

    results.sort(key=lambda r: r["grams"], reverse=True)
    summary = summarize_results(results)
    return summary, results, rows


def analyze_bgr_frame(frame_bgr: np.ndarray, models: dict, save_prefix: str = None):
    """
    Convenience wrapper if you have a BGR numpy frame (cv2 / Picamera2).
    """
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return analyze_pil(pil, models, save_prefix=save_prefix)


if __name__ == "__main__":
    main()

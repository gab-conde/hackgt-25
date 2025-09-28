# waste_estimator_full.py
import os, io, time, math, requests
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ===================== CONFIG =====================
IMG_SRC = "/Users/albertzheng/Downloads/plate_nutter.jpg"  # <-- set your image path

# Prompts for OWL-ViT (you can tweak/extend)
LABELS = [
    "plate", "dining plate", "dish",
    "a small round peanut butter cookie",
    "a cookie", "a round cookie", "a biscuit",
    "peanut butter sandwich cookie"
]
THRESH = 0.20

# Known plate diameter: 8.5 in = 21.6 cm
PLATE_DIAM_CM = 21.6

# Clamp heights to tame monocular depth noise (meters)
H_MIN_M = 0.000
H_MAX_M = 0.020  # 2 cm cap for small snacks; raise to 0.06–0.08 for mounded foods

# Rough volumetric densities (g/cm^3). Add more as needed.
RHO_TABLE = {
    "beef": 1.00, "grilled beef": 1.00, "short ribs": 1.00, "meat": 1.00,
    "rice": 0.95, "kimchi": 0.60, "lettuce": 0.15, "banchan": 0.70,
    "cookie": 0.70, "biscuit": 0.65, "cracker": 0.55
}

# ==================================================

def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        data = requests.get(path_or_url, timeout=15).content
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img

@torch.no_grad()
def owlvit_detect(image: Image.Image, labels, thresh=0.2, device="cpu"):
    """Run OWL-ViT and return list of dicts: {box:[x1,y1,x2,y2], score, label}"""
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device).eval()
    inputs = processor(text=[labels], images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    outputs = model(**inputs)
    target_sizes = torch.tensor([(image.height, image.width)], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=thresh, text_labels=[labels]
    )[0]

    dets = []
    for box, score, lab in zip(results["boxes"].detach().cpu().tolist(),
                               results["scores"].detach().cpu().tolist(),
                               results["text_labels"]):
        dets.append({"box":[int(x) for x in box], "score":float(score), "label":lab})
    return dets

def draw_dets(image, dets, plate_box=None):
    img = image.copy()
    drw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()
    for d in dets:
        x1,y1,x2,y2 = d["box"]
        is_plate = d["label"].lower() in ["plate","dining plate","dish"]
        color = "cyan" if is_plate else "lime"
        drw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        drw.text((x1+3, max(0,y1-18)), f"{d['label']} {d['score']:.2f}", fill=color, font=font)
    if plate_box:
        drw.rectangle(plate_box, outline="red", width=3)
        drw.text((plate_box[0], max(0, plate_box[1]-18)), "plate(scale)", fill="red", font=font)
    return img

def bbox_to_circle(bbox):
    """Approximate a circle (cx,cy,r) from a bbox (x1,y1,x2,y2)"""
    x1,y1,x2,y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r  = int(0.5 * min(x2-x1, y2-y1))
    return (cx, cy, r)

def get_plate_scale_and_circle(image_pil, dets, plate_diam_cm=PLATE_DIAM_CM):
    """
    Find plate and cm/px.
    1) Try OWL-ViT 'plate' → bbox → circle approximation.
    2) Fallback to OpenCV HoughCircles (returns true circle).
    Returns: (plate_box, cm_per_px, circle_tuple or None)
    """
    for d in dets:
        if d["label"].lower() in ["plate","dining plate","dish"]:
            x1,y1,x2,y2 = d["box"]
            plate_px = max(x2 - x1, y2 - y1)
            cm_per_px = plate_diam_cm / max(plate_px, 1)
            circ = bbox_to_circle(d["box"])
            print(f"[INFO] Plate via OWL-ViT, cm/px = {cm_per_px:.4f}")
            return d["box"], cm_per_px, circ

    # Fallback: HoughCircles
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
        param1=100, param2=40, minRadius=80, maxRadius=1000
    )
    if circles is not None:
        x, y, r = np.round(circles[0,0]).astype("int")
        plate_px = 2 * r
        cm_per_px = plate_diam_cm / max(plate_px, 1)
        bbox = (x-r, y-r, x+r, y+r)
        print(f"[INFO] Plate via OpenCV HoughCircles, cm/px = {cm_per_px:.4f}")
        return bbox, cm_per_px, (x, y, r)

    print("[WARN] Plate not detected. Using fallback scale 0.05 cm/px (relative only).")
    return None, 0.05, None

# -------- ZoeDepth loader (cached, with fallbacks) --------
_ZOE = None
@torch.no_grad()
def get_zoe(device="cpu"):
    """
    Load ZoeDepth via torch.hub with variant fallbacks.
    Ensure your env has timm==0.6.13 to avoid state_dict mismatches.
    """
    global _ZOE
    if _ZOE is not None:
        return _ZOE
    variants = ["ZoeD_N", "ZoeD_NK", "ZoeD_K"]
    last_err = None
    for v in variants:
        try:
            print(f"[INFO] Loading ZoeDepth: {v}")
            m = torch.hub.load("isl-org/ZoeDepth", v, pretrained=True, trust_repo=True)
            _ZOE = m.to(device).eval()
            print(f"[INFO] Loaded {v}")
            return _ZOE
        except Exception as e:
            print(f"[WARN] {v} failed: {e}")
            last_err = e
    raise RuntimeError(f"All Zoe variants failed. Last error: {last_err}")

@torch.no_grad()
def zoe_infer_depth(image_pil, device="cpu"):
    model = get_zoe(device)
    return model.infer_pil(image_pil)  # HxW numpy (meters)

# ---------- Plane from plate & global height map ----------
def height_map_from_plate(depth_m, plate_circle, cm_per_px, h_min=H_MIN_M, h_max=H_MAX_M):
    """
    Fit plane z = a x + b y + c on the **inner 90% of plate** and
    return height map (meters) above that plane for the whole image.
    """
    H, W = depth_m.shape
    cx, cy, r = plate_circle
    # Smoothing helps ZoeDepth noise
    depth_sm = cv2.medianBlur(depth_m.astype(np.float32), 3)

    yy, xx = np.mgrid[0:H, 0:W]
    plate_mask = (xx - cx)**2 + (yy - cy)**2 <= int((r * 0.90)**2)

    ys, xs = np.nonzero(plate_mask)
    if len(xs) < 500:
        # Extreme fallback: use global median as plane
        z_bg = np.median(depth_sm)
        h_m = np.clip(z_bg - depth_sm, h_min, h_max)
        return h_m, plate_mask

    xs_m = (xs * cm_per_px) / 100.0
    ys_m = (ys * cm_per_px) / 100.0
    Z    = depth_sm[ys, xs]
    A = np.c_[xs_m, ys_m, np.ones_like(xs_m)]
    a, b, c = np.linalg.lstsq(A, Z, rcond=None)[0]  # z = a x + b y + c

    Z_plane = a*((xx*cm_per_px)/100.0) + b*((yy*cm_per_px)/100.0) + c
    h_m = np.clip(Z_plane - depth_sm, h_min, h_max)

    # Debug: median height on the plate should be near 0
    plate_bg_h_mm = float(np.median(h_m[plate_mask]) * 1000.0)
    print(f"[DEBUG] plate median height ~ {plate_bg_h_mm:.2f} mm (should be ~0)")
    return h_m, plate_mask

def integrate_volume_cm3_over_box(h_m, roi_box, cm_per_px):
    x1,y1,x2,y2 = roi_box
    H, W = h_m.shape
    x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
    H_roi = h_m[y1:y2+1, x1:x2+1]
    pixel_area_cm2 = (cm_per_px ** 2)
    vol_cm3 = float(H_roi.sum() * 100.0 * pixel_area_cm2)  # meters → cm (×100)
    return max(0.0, vol_cm3)

def label_to_rho(label: str) -> float:
    """Map free-form labels to a density."""
    lab = label.lower()
    # direct hit
    if lab in RHO_TABLE:
        return RHO_TABLE[lab]
    # contains keywords
    for k in ["cookie", "biscuit", "cracker"]:
        if k in lab:
            return RHO_TABLE[k]
    for k in ["beef", "meat", "rice", "kimchi", "lettuce", "banchan"]:
        if k in lab:
            return RHO_TABLE[k]
    return 0.80  # safe fallback

def estimate_grams(volume_cm3, label):
    rho = label_to_rho(label)
    return max(1.0, min(volume_cm3 * rho, 1500.0))

# ===================== MAIN =====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    image = load_image(IMG_SRC)

    # 1) OWL-ViT
    dets = owlvit_detect(image, LABELS, thresh=THRESH, device=device)
    print(f"[INFO] {len(dets)} detections")
    for d in dets:
        print(f"  - {d['label']}  {d['score']:.2f}  {d['box']}")

    # 2) Plate scale + circle (OWL-ViT → Hough fallback)
    plate_box, cm_per_px, plate_circle = get_plate_scale_and_circle(image, dets)
    if plate_circle is None and plate_box is not None:
        plate_circle = bbox_to_circle(plate_box)

    # 3) ZoeDepth once
    t0 = time.time()
    depth_m = zoe_infer_depth(image, device=device)
    print(f"[INFO] ZoeDepth took {(time.time()-t0)*1000:.1f} ms")

    # 4) Global height map from the **plate plane**
    if plate_circle is None:
        print("[WARN] No plate circle; using global median plane (less accurate).")
        # synthetic circle covering center to keep flow
        H, W = depth_m.shape
        plate_circle = (W//2, H//2, min(W,H)//3)
    h_m, plate_mask = height_map_from_plate(depth_m, plate_circle, cm_per_px, H_MIN_M, H_MAX_M)

    # 5) For each non-plate detection, estimate volume + grams
    results = []
    for d in dets:
        if d["label"].lower() in ["plate","dining plate","dish"]:
            continue
        roi = d["box"]
        vol = integrate_volume_cm3_over_box(h_m, roi, cm_per_px)
        g   = estimate_grams(vol, d["label"])
        results.append({"label": d["label"], "score": d["score"], "box": roi,
                        "volume_cm3": vol, "grams": g})

    # Sort by grams desc for display
    results.sort(key=lambda r: r["grams"], reverse=True)
    for r in results:
        print(f"[RESULT] {r['label']:<30}  vol≈{r['volume_cm3']:7.1f} cm^3   wt≈{r['grams']:6.1f} g   box={r['box']}")

    # 6) Save annotated image
    vis = draw_dets(image, dets, plate_box)
    out_path = os.path.splitext(IMG_SRC)[0] + "_annotated.jpg"
    vis.save(out_path, quality=95)
    print(f"[INFO] Saved annotated image to {out_path}")

if __name__ == "__main__":
    main()

import time, io, os, requests
import numpy as np
import torch
import cv2
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from depth_full import *


# path to image
IMG_PATH = '~/Downloads/plate.'

# configure device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize camera
camera = Picamera2()
camera_config = camera.create_preview_configuration()
camera.configure(camera_config)

# start camera
camera.start()

# allow camera time to stabilize
time.sleep(2)

# capture images
for i in range(2):  # do 2 images for demo purposes
    # capture frame as numpy arrary
    frame = camera.capture_array("main")
    print(frame.shape)

    # encode image to jpg format
    # success_flag, buffer = cv2.imencode('.jpg', frame)

    # use to write file for testing
    # cv2.imwrite(f"image{i}.jpg", frame)

    image = Image.from_array(frame)

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
    out_path = os.path.splitext(IMG_PATH)[0] + "_annotated.jpg"
    vis.save(out_path, quality=95)
    print(f"[INFO] Saved annotated image to {out_path}")


    # wait for new plate to enter frame - 10 seconds
    time.sleep(10)
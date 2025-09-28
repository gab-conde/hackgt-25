# app.py (sketch)
import time, cv2
from picamera2 import Picamera2
from full_pipeline_gdino_sam_da2 import init_pipeline, analyze_bgr_frame

# init models once
models = init_pipeline()  # or init_pipeline(device="cpu")

cam = Picamera2()
cfg = cam.create_preview_configuration()
cam.configure(cfg)
cam.start()
time.sleep(2)

for i in range(2):
    frame = cam.capture_array("main")            # BGR numpy array
    # (optional) fix color if needed:
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # only if your stream is RGBA

    summary, results, rows = analyze_bgr_frame(frame, models, save_prefix=None)
    # 'summary' is exactly: [{name, volume_cm3, grams}, ...]

    print("[FINAL]", summary)  # e.g. [{'name': 'hamburger bun', 'volume_cm3': 226.3, 'grams': 181.0}]

    # wait for next plate
    time.sleep(10)

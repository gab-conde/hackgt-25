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
    frame = camera.capture_array("main")          # RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results, rows = analyze_bgr_frame(frame_bgr, models, save_prefix=f"/home/pi/frame_{i}")
    # analyze_bgr_frame now pushes to API via send_disposal_batch()
    time.sleep(10)

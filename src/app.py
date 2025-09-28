import sys, os
sys.path.append("/home/eli/Desktop/hackgt25/Depth-Anything-V2/depth_anything_v2")

import time, cv2, torch
import numpy as np
import matplotlib
from picamera2 import Picamera2

da_path = "/home/eli/Desktop/hackgt25/Depth-Anything-V2"
if os.path.isdir(da_path) and da_path not in sys.path:
    sys.path.insert(0, da_path)

from depth_anything_v2.dpt import DepthAnythingV2



# ==================== Depth Anything Setup ==================== #

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# using small model
encoder ='vits'

# configure the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'../../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


# ==================== Rasberry Pi Camera Setup ==================== #

# initialize camera
camera = Picamera2()
camera_config = camera.create_preview_configuration()
camera.configure(camera_config)

# start camera
camera.start()

# allow camera time to stabilize
time.sleep(2)

# ==================== Image Processing ==================== #

for i in range(2):  # do 2 images for demo purposes
    # capture frame as numpy arrary
    frame = camera.capture_array("main")
    # correct the color map
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # print(frame.shape)

    # encode image to jpg format
    # success_flag, buffer = cv2.imencode('.jpg', frame)

    # use to write file for testing
    cv2.imwrite(f"image{i}.jpg", frame)

    # produce depth map from image (HxW in numpy)
    depth = model.infer_image(frame_rgb) # HxW raw depth map in numpy

    # THE FOLLOWING LINES FOR VISUALIZING ONLY
    # scale depth map to range 0 to 255
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    # drop alpha channel, convert from rgb to bgr, conver to uint8 for viewing
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(f"image{i}_depth.jpg", depth)

    # wait for new plate to enter frame - 10 seconds
    time.sleep(2)
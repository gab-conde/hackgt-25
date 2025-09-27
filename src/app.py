import time
import cv2
from picamera2 import Picamera2

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

    # wait for new plate to enter frame - 10 seconds
    time.sleep(10)
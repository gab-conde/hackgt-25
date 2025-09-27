# Libraries for PI
from flask import Flask, Response
from picamera2 import Picamera2
import time
import cv2

# Create Flask App
app = Flask(__name__)

# Capture Defualt Camera  and use defualt settings
picamera  = Picamera2()
camera_config = picamera.create_preview_configuration()
picamera.configure(camera_config)
picamera.start()

# For stabalization
time.sleep(2)

def generate_frames():
    while True:
        frame = picamera.capture_array()
        # Convert Frame to jpeg to turn to MJPEG
        _,buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()

        # Get Frame in byte format for MJPEG
        yield (b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1> Raspberry PI Camera Feed</h1><img src="/video">'

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=False)
from picamera import Picamera
from io import BytesIO

stream = BytesIO()
# sensor_mode=4 : 1640x1232, 4:3, <= 40fps, full FoV
# https://picamera.readthedocs.io/en/release-1.12/fov.html#camera-modes
camera = PiCamera(resolution=(1640, 1232), framerate=40, sensor_mode=4)

print('Started recording')
camera.start_recording(stream, format='rgb')
camera.wait_recording(5)
camera.stop_recording()
print('Stopped recording')
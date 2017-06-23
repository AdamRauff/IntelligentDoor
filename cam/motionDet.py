from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import datetime
import imutils

minArea = 500

firstFrame = None
# initialize camera
camera = PiCamera()

# decrease camera resolution and frame rate for faster processing
camera.resolution = (640, 480)
camera.framerate = 32

# grab a reference to the raw camera capture
rawCapture = PiRGBArray(camera, size=(640,480))

# allow camera to warmup
time.sleep(0.1)

# grab frames from camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
	# grab raw numpy array
	image = frame.array

	# display image
	cv2.imshow("Frame", image)

	# wait for a second
	key = cv2.waitKey(1) & 0xFF

	# clear stream for next frame
	rawCapture.truncate(0)

	# quit when q is pressed
	if key == ord("q"):
		break

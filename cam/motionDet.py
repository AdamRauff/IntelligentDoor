# Adam Rauff
# 6/25/17
# This file originates from the blog post of pyimagesearch
# the purpose is to supply the smart door with ability to detect people outside door, and dislay video stream on the interir mirror

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import datetime
import imutils
import json

minArea = 300

firstFrame = None

# initialize camera
camera = PiCamera()

# decrease camera resolution and frame rate for faster processing
camera.resolution = (640, 480)
camera.framerate = 32

# grab a reference to the raw camera capture
rawCapture = PiRGBArray(camera, size=(640,480))

# allow camera to warmup
time.sleep(2)

# grab frames from camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    # grab raw numpy array
    image = frame.array
        
    # set variable text to remark no objects detected
    text = "No Objects"

    # resize image to be of width 500 pixels while keeping 
    # aspect ration the same
    image = imutils.resize(image, width=500)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply blur
    gray = cv2.GaussianBlur(gray, (21,21),0)

    if firstFrame is None:
        firstFrame = gray

        # truncate stream
        rawCapture.truncate(0)

        # if first frame, go on to first image to be processed
        continue
        
    # compute difference between current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)

    # apply a threshold to diff image (discard very slight changes
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # apply a small erosion followed by a slightly bigger dialtion
    # erosion is done to git rid of small speckles
    # dilation is done to offset the pixels eroded of the remaining 
    # pixels, and highlight areas of movement in image
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=3)

    # find contours
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, \
            cv2.CHAIN_APPROX_SIMPLE)

    # loop over contours
    for c in cnts:
        # ignore small contours
        if cv2.contourArea(c) < minArea:
            continue

        # compute bounding box for contour
        (x,y,w,h) = cv2.boundingRect(c)
        
        # draw box on frame
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,245,0),2)

        # change text
        text = 'Motion'

    # draw text and timestamp on frame
    cv2.putText(image, "Status: {}".format(text), (10,20), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    cv2.putText(image, datetime.datetime.now().strftime( \
            "%A %d %B %Y %I:%M:%S%p"),(10,image.shape[0]-10), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),1)

    # display images
    cv2.imshow("Securtiy Feed", image)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    # wait for a second
    key = cv2.waitKey(1) & 0xFF

    # clear stream for next frame
    rawCapture.truncate(0)

    # quit when q is pressed
    if key == ord("q"):
            break

cv2.destroyAllWindows()

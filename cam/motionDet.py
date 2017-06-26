# Adam Rauff
# 6/25/17
# This file originates from the blog post of pyimagesearch
# the purpose is to supply the smart door with ability to detect people outside door, and dislay video stream on the interir mirror
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import datetime
import imutils
import json
import warnings

# provide path to json file, could make this input 
# through command line with argparse
jpath = '/home/pi/SmDr/cam/conf.json'

# filter warnings when loading json
warnings.filterwarnings('ignore')

# load configuration
conf = json.load(open(jpath))

# initialize camera
camera = PiCamera()

# decrease camera resolution and frame rate for faster processing
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]

# grab a reference to the raw camera capture
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow camera to warmup
print('warming up ...')
time.sleep(conf["camera_warmup_time"])

# initialize average frame, last timestamp, and frame motion counter
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

minArea = conf["min_area"] 

# grab frames from camera
for f in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    # grab raw numpy array
    image = f.array
    
    timestamp = datetime.datetime.now()

    # set variable text to remark no objects detected
    text = "No Objects"

    # resize image to be of width 500 pixels while keeping 
    # aspect ration the same
    image = imutils.resize(image, width=500)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply blur
    gray = cv2.GaussianBlur(gray, (21,21),0)

    if avg is None:
        print('starting background model ...') 
        avg = gray.copy().astype("float")
        BlcIM = gray.copy()
        BlcIM[np.where(BlcIM > 0)] = 0
        # construct window to display images
        cv2.namedWindow('Security Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('Security Feed', BlcIM)
        # truncate stream
        rawCapture.truncate(0)
        # if first frame, go on to first image to be processed
        continue
    
    # accumulate weighted average of current and previous frame
    cv2.accumulateWeighted(gray, avg, 0.5)
    # compute difference between current frame and first frame
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # apply a threshold to diff image (discard very slight changes
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, \
            cv2.THRESH_BINARY)[1]

    # apply a small erosion followed by a slightly bigger dialtion
    # erosion is done to git rid of small speckles
    # dilation is done to offset the pixels eroded of the remaining 
    # pixels, and highlight areas of movement in image
    # thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, \
            cv2.CHAIN_APPROX_SIMPLE)

    # loop over contours
    for c in cnts:
        # ignore small contours
        if cv2.contourArea(c) < minArea:
            continue
        # compute bounding box for contour
        # (x,y,w,h) = cv2.boundingRect(c)
        # draw box on frame
        # cv2.rectangle(image, (x,y), (x+w, y+h), (0,245,0),2)

        # change text
        text = 'Motion'
   
    # see if motion is detected
    if text == 'Motion':
        # print('Entered Motion if')

        # check if enough time has passed
        if (timestamp-lastUploaded).seconds >= conf["min_upload_seconds"]:

            # increment motion counter
            motionCounter += 1

            if motionCounter >= conf["min_motion_frames"]:
                # print('Entered MotionCount if')

                # draw text and timestamp on frame
                cv2.putText(image, "Status: {}".format(text), (10,20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
                ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
                cv2.putText(image,ts,(10,image.shape[0]-10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),1)
    
                if conf["show_video"]:
                    
                    # display images
                    cv2.imshow("Security Feed", image)
                    # cv2.imshow("Thresh", thresh)
                    # cv2.imshow("Frame Delta", frameDelta)

                    # wait for a second
                    key = cv2.waitKey(1) & 0xFF
                    
                    # quit when q is pressed
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break

                # update last uploaded timestamp and reset motion counter
                lastUploaded = timestamp
                motionCounter = 0
    else:


        # display black image
        cv2.imshow("Security Feed", BlcIM)
        
        motionCounter = 0

        # wait for a second
        key = cv2.waitKey(1) & 0xFF
        
        # quit when q is pressed
        if key == ord("q"):
            break

    # clear stream for next frame
    rawCapture.truncate(0)

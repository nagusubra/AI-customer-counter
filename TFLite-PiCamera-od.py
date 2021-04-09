import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


class VideoStream:
    """
    
    Camera object that controls video streaming from the Picamera
    Define VideoStream class to handle streaming of video from webcam in separate processing thread
    Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
    
    """
    def __init__(self, resolution=(640,480),framerate=60):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture('videos/test-video.h264')
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame
    
    def stop(self):
        """Destroy the root of the object and release all resources"""
        print ("[Info] CLosing ...")
        self.stopped = True

    def destructor(self):
        """Destroy the root of the object and release all resources"""
        print ("[Info] CLosing ...")
        self.stopped = True
            
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='models/model1.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='models/labels.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH.',
                    default='1280x720')
args = parser.parse_args()
PATH_TO_MODEL_DIR = args.model
PATH_TO_LABELS = args.labels 
MIN_CONF_THRESH = float(args.threshold)


resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
import time
print('[Info] Loading model...', end='')
start_time = time.time()


interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('[Info] Done! Took {} seconds'.format(elapsed_time))

interpreter.allocate_tensors()
input_details1 = interpreter.get_input_details()
input_details2 = interpreter.get_input_details()
output_details1 = interpreter.get_output_details()
output_details2 = interpreter.get_output_details()

height = input_details1[0]['shape'][1]
width = input_details1[0]['shape'][2]

#print ("Height : " + str(height) + "  Width : " + str(width))

floating_model = (input_details1[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5           #Both the mean add up to 255 so that all the pixel are colour corrected for analyising the frames


frame_rate_calc = 1#Initialize video stream
freq = cv2.getTickFrequency()
print('[Info] Running inference for PiCamera')

videostream = VideoStream(resolution=(imW,imH),framerate=30).start() #"""Initialize video stream"""
time.sleep(1)


cnt_enter = 0
cnt_exit = 0

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    current_count=0                                                 #"""Start timer (for calculating frame rate)"""
    t1 = cv2.getTickCount()
    frame1 = videostream.read()                                     #"""Grab frame from video stream """
    frame = frame1.copy()                                           #"""Acquire frame and resize to expected shape [1xHxWx3]"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    frame_rgb1 = frame_rgb[0:720, 0:600]
    frame_resized1 = cv2.resize(frame_rgb1, (width, height))
    input_data1 = np.expand_dims(frame_resized1, axis=0)
    if floating_model:                                              #"""Normalize pixel values if using a floating model (i.e. if model is non-quantized) Input normalization is a common technique in machine learning. This specific model was trained with input value range -1 to 1, so we should normalize the inference input to the same range to achieve best result."""
        input_data1 = (np.float32(input_data1) - input_mean) / input_std

    interpreter.set_tensor(input_details1[0]['index'],input_data1)    #""" Perform the actual detection by running the model with the image as input """
    interpreter.invoke()


    # Retrieve detection results
    boxes1 = interpreter.get_tensor(output_details1[0]['index'])[0] # Bounding box coordinates of detected objects
    classes1 = interpreter.get_tensor(output_details1[1]['index'])[0] # Class index of detected objects
    scores1 = interpreter.get_tensor(output_details1[2]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores1)):
        if classes1[i].all() == 0.0:

            if ((scores1[i] > MIN_CONF_THRESH) and (scores1[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes1[i][0] * imH)))
                xmin = int(max(1,(boxes1[i][1] * imW/2.1)))
                ymax = int(min(imH,(boxes1[i][2] * imH)))
                xmax = int(min(imW,(boxes1[i][3] * imW/2.1)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes1[i])] # Look up object name from "labels" array using class index
                
                if object_name == 'person':
                    label = '%s: %d%%' % ('Customer', int(scores1[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    cnt_exit+=1

    
    frame_rgb2 = frame_rgb[0:720, 680:1280]
    frame_resized2 = cv2.resize(frame_rgb2, (width, height))
    input_data2 = np.expand_dims(frame_resized2, axis=0)
    if floating_model:                                              #"""Normalize pixel values if using a floating model (i.e. if model is non-quantized) Input normalization is a common technique in machine learning. This specific model was trained with input value range -1 to 1, so we should normalize the inference input to the same range to achieve best result."""
        input_data2 = (np.float32(input_data2) - input_mean) / input_std

    interpreter.set_tensor(input_details2[0]['index'],input_data2)    #""" Perform the actual detection by running the model with the image as input """
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details2[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details2[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details2[2]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if classes[i].all() == 0.0:

            if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(0,((boxes[i][1] * imW/2)+680)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                
                if object_name == 'person':
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    cnt_enter+=1



    cv2.putText(frame,'Frame rate per sec: {0:.2f}'.format(frame_rate_calc),(15,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.line(frame, (640,0), (640,720), (0,250,250), 5)

    cv2.putText (frame,'Entered : '+ str(cnt_enter),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText (frame,'Exit : '+ str(cnt_exit),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)

    cv2.putText (frame,'Customer Counter, Welcome to UofC!',(350,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
    #cv2.putText (frame,'Total Customer Count : ' + str(current_count),(450,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object Detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
print("Done")

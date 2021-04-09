import picamera     # Importing the library for camera module
import os
import os.path
from time import sleep  # Importing sleep from time library to add delay in program

completed_video = '/home/pi/ENEL400/AI-customer-counter/videos/test-video.h264'
filename = 'test-video'
camera = picamera.PiCamera()    # Setting up the camera
camera.start_preview()      # You will see a preview window while recording
camera.start_recording(completed_video) # Video will be saved at desktop
sleep(10)
camera.stop_recording()
camera.stop_preview()

#command = "MP4Box -add {} {}.mp4".format(completed_video, os.path.splitext(filename)[0])
#try:
#    output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
#except subprocess.CalledProcessError as e:
#    print('FAIL:\ncmd:{}\noutput:{}'.format(e.cmd, e.output))
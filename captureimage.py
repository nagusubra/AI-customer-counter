import picamera     # Importing the library for camera module
from time import sleep
camera = picamera.PiCamera()    # Setting up the camera
camera.start_preview()
sleep(5)
camera.capture('/home/pi/ENEL400/AI-customer-counter/images/test-image.jpg') # Capturing the image
camera.stop_preview()
print('Done')
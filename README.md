# AI customer counter

A project for ENEL 400, based on image detection using Tensor flow on Raspberry Pi which uses a camera to detect the number of people entering and exiting a store.

The 3 stages of the project:

1. https://www.youtube.com/watch?v=xjZYEzk6qSo&t=3s
2. https://www.youtube.com/watch?v=cgwiGRqhjds&t=63s
3. https://www.youtube.com/watch?v=8usop44kELs&t=5s


## Steps to run the code 
1. Setting up the Raspberry Pi and Getting Updates
2. Organizing our Workspace and Virtual Environment
3. Installing the Prerequisites
4. Running Object Detection on Image, Video, or Pi Camera

## Step 1: Setting up the Raspberry Pi and Getting Updates
Before we can get started, we must have access to the Raspberry Pi's Desktop Interface. This can be done with VNC Viewer or the standard Monitor and HDMI. 

Once you have access to the Desktop Interface, either remote or physical, open up a terminal. Retrieve updates for the Raspberry Pi with

```
sudo apt-get update
sudo apt-get dist-upgrade
```

Depending on how recently you setup or updated your Pi, this can be instantaneous or lengthy. After your Raspberry Pi is up-to-date, we should make sure our Camera is enabled. First to open up the System Interface, use

```
sudo raspi-config
```

Then navigate to Interfacing Options -> Camera and make sure it is enabled. Then hit Finish and reboot if necessary.

<p align="left">
  <img src="doc/Camera Interface.png">
</p>

## Step 2: Organizing our Workspace and Virtual Environment

Then, your going to want to clone this repository with

```
git clone https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi.git
```

This name is a bit long so let's trim it down with

```
mv TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi tensorflow
```

We are now going to create a Virtual Environment to avoid version conflicts with previously installed packages on the Raspberry Pi. First, let's install virtual env with

```
sudo pip3 install virtualenv
```

Now, we can create our ```tensorflow``` virtual environment with

```
python3 -m venv tensorflow
```

There should now be a ```bin``` folder inside of our ```tensorflow``` directory. So let's change directories with

```
cd tensorflow
```

We can then activate our Virtual Envvironment with

```
source bin/activate
```

**Note: Now that we have a virtual environment, everytime you start a new terminal, you will no longer be in the virtual environment. You can reactivate it manually or issue ```echo "source tensorflow/bin/activate" >> ~/.bashrc```. This basically activates our Virtual Environment as soon as we open a new terminal. You can tell if the Virtual Environment is active by the name showing up in parenthesis next to the working directory.**

When you issue ```ls```, your ```tensorflow``` directory should now look something like this

<p align="left">
  <img src="doc/directory.png">
</p>

## Step 3: Installing the Prerequisites


(https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)


This step should be relatively simple. I have compressed all the commands into one shellscript which you can run with
```
bash install-prerequisites.sh
```
This might take a few minutes, but after everything has been installed you should get this message
```
Prerequisites Installed Successfully
```
Now, it's best to test our installation of the tflite_runtime module. To do this first type
```
python
```
From the Python terminal, enter these lines
```
Python 3.7.3 (default, Jul 25 2020, 13:03:44)
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tflite_runtime as tf
>>> print(tf.__version__)
```
If everything went according to plan, you should get 
```
2.5.0
```

## Step 4: Running Object Detection on Image, Video, or Pi Camera
Now we're finally ready to run object detection! In this repository, I've included a sample model that I converted as well as 3 object detection scripts utilizing OpenCV. If you want to convert a custom model or a pre-trained model, you can look at the [TensorFlow Lite Conversion Guide](https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-Conversion.md) that I wrote. A majority of the code is from [Edje Electronics' tutorial](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi) with a few small adjustments. 
- ```TFLite-PiCamera-od.py```: This program uses the Pi Camera to perform object detection. It also counts and displays the number of objects detected in the frame. The usage is
```
usage: TFLite-PiCamera-od.py [-h] [--model MODEL] [--labels LABELS]
                             [--threshold THRESHOLD] [--resolution RESOLUTION]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Provide the path to the TFLite file, default is
                        models/model.tflite
  --labels LABELS       Provide the path to the Labels, default is
                        models/labels.txt
  --threshold THRESHOLD
                        Minimum confidence threshold for displaying detected
                        objects
  --resolution RESOLUTION
                        Desired webcam resolution in WxH. If the webcam does
                        not support the resolution entered, errors may occur.
```
- ```TFLite-Image-od.py```: This program takes a single image as input. I haven't managed to get it to run inference on a directory yet. If you do, feel free to share it as it might help others. This script also counts the number of detections. The usage is
```
usage: TFLite-Image-od.py [-h] [--model MODEL] [--labels LABELS]
                          [--image IMAGE] [--threshold THRESHOLD]
                          [--resolution RESOLUTION]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Provide the path to the TFLite file, default is
                        models/model.tflite
  --labels LABELS       Provide the path to the Labels, default is
                        models/labels.txt
  --image IMAGE         Name of the single image to perform detection on
  --threshold THRESHOLD
                        Minimum confidence threshold for displaying detected
                        objects
  --resolution RESOLUTION
                        Desired webcam resolution in WxH. If the webcam does
                        not support the resolution entered, errors may occur.
 ```
- ```TFLite-Video-od.py```: This program is similar to the last two however it takes a video as input. The usage is
```
usage: TFLite-Video-od.py [-h] [--model MODEL] [--labels LABELS]
                          [--video VIDEO] [--threshold THRESHOLD]
                          [--resolution RESOLUTION]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Provide the path to the TFLite file, default is
                        models/model.tflite
  --labels LABELS       Provide the path to the Labels, default is
                        models/labels.txt
  --video VIDEO         Name of the video to perform detection on
  --threshold THRESHOLD
                        Minimum confidence threshold for displaying detected
                        objects
  --resolution RESOLUTION
                        Desired webcam resolution in WxH. If the webcam does
                        not support the resolution entered, errors may occur.
 ```
 
 Now to finally test it out, run
 ```
 python TFLite-PiCamera-od.py
 ```
 
 

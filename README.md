# ObjectDetection
An yoloV3 model used with OpenCv to detect object through webcam.
![Screenshot (75)](https://user-images.githubusercontent.com/102272183/213876079-7a187414-c8aa-4a6c-80fc-a3de0feb65cc.png)
![Screenshot (74)](https://user-images.githubusercontent.com/102272183/213876187-a48e4bde-c7ee-49d3-9115-6ce2c77d84d0.png)

## Object Detection using YOLO v3
Introduction
This code implements object detection using YOLO v3 (You Only Look Once) algorithm. YOLO is a popular real-time object detection system that can detect multiple objects in a single image or video frame.

## Requirements
To run this code, you need to install the following libraries:

OpenCV (cv2)

Numpy

Matplotlib

## You also need to download the following files:

yolov3.weights

yolov3.cfg

coco.names

## Code Explanation
The code starts by importing the necessary libraries and reading the YOLO model using the OpenCV function cv2.dnn.readNet.

The classes variable is then initialized as an empty list, which is later filled with the names of the classes that the YOLO model can detect. This list is read from the coco.names file.

Next, the code captures video from the default camera using OpenCV's cv2.VideoCapture function. The video frames are then processed one by one in a while loop.

For each frame, the code performs the following steps:

The frame is passed through the YOLO model to obtain the output.

The output is processed to extract the bounding boxes, confidence scores, and class IDs of the objects detected in the frame.

Non-maximum suppression (NMS) is applied to the bounding boxes to eliminate overlapping boxes and keep only the most confident detection for each object.

The remaining bounding boxes are drawn on the frame along with the labels indicating the class of the objects and the confidence scores.

The processed frame is displayed on the screen.

The loop continues until the user presses the 'x' key.

## Conclusion
This code demonstrates how to implement object detection using YOLO v3. It can be used as a starting point for developing more complex computer vision applications that involve object detection.



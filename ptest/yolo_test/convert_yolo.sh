#!/usr/bin/env bash

pip install ultralytics

# object detection
yolo export model=models/yolo11n.pt format=torchscript
yolo export model=models/yolo11n.pt format=onnx

# instance segmentation
yolo export model=models/yolo11n-seg.pt format=torchscript
yolo export model=models/yolo11n-seg.pt format=onnx

# pose detection
yolo export model=models/yolo11n-pose.pt format=torchscript
yolo export model=models/yolo11n-pose.pt format=onnx

import argparse
import sys

import aria.sdk as aria

import cv2
import numpy as np


from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord


import math

import cv2
from ultralytics import YOLO

from utils import *

def init_yolo():
    model = YOLO("yolov8n.pt")
    classNames = yolo_class_names()
    return model, classNames

def draw_bounding_boxes(results, img, classNames):
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(
                img, classNames[int(box.cls[0])], [x1, y1], 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
    
    return img
    
from ariaStreaming import AriaStreaming            

def main():
    args = parse_args()
    
        
    model, class_names = init_yolo()

    aria_streaming = AriaStreaming(args)
    aria_streaming.init_cv2_windows()
    
    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if aria_streaming.observer.img is not None:
                img = cv2.cvtColor(aria_streaming.observer.img, cv2.COLOR_BGR2RGB)
                
                if args.correct_distorsion:
                    img = distort_by_calibration(
                        img, aria_streaming.dst_calib, aria_streaming.rgb_calib
                    )
                
                results = model(img, stream=True, verbose=False)

                img = draw_bounding_boxes(results, img, class_names)
                cv2.imshow(aria_streaming.window, np.rot90(img, -1))

                

                aria_streaming.observer.img = None

    aria_streaming.unsubscribe()
    


if __name__ == "__main__":
    main()

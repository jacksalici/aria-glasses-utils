import aria.sdk as aria


from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

import math
import cv2
import torch
import numpy as np


from utils import *

from ariaStreaming import AriaStreaming            
from eyeGaze import EyeGaze
from gazeInference import GazeInference


def main():
    args = parse_args()

    aria_streaming = AriaStreaming(args, cameras = "RGB-ET")
    aria_streaming.init_cv2_windows("RGB")
    aria_streaming.init_cv2_windows("ET")
    
    eye_gaze = EyeGaze(live=True, calib_live = aria_streaming.sensors_calib)
    gaze_inference = GazeInference()
    gaze_center_in_pixels = [0,0]
    
    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if aria.CameraId.Rgb in aria_streaming.observer.imgs:
                rgb_image = np.rot90(aria_streaming.observer.imgs[aria.CameraId.Rgb], -1).copy()
                print(gaze_center_in_pixels)
                rgb_image = cv2.circle(rgb_image, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 4)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                cv2.imshow('RGB', rgb_image)
                del aria_streaming.observer.imgs[aria.CameraId.Rgb]
                
            if aria.CameraId.EyeTrack in aria_streaming.observer.imgs:
                image = aria_streaming.observer.imgs[aria.CameraId.EyeTrack]
                cv2.imshow('ET', image)
                img = torch.tensor(image, device='cpu')
                yaw, pitch = gaze_inference.predict(img)
                gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center_raw(yaw, pitch)
                del aria_streaming.observer.imgs[aria.CameraId.EyeTrack]

                    


    aria_streaming.unsubscribe()

if __name__ == "__main__":
    main()
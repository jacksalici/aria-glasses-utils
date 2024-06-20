#general libraries
import os
import shutil
import cv2
import numpy as np
import tomllib
import torch
from enum import Enum
import typing

#project aria libraries
from projectaria_tools.core import data_provider, image, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

#project classes
from utils import *
from eyeGaze import EyeGaze
from gazeInference import GazeInference



class Streams(Enum):
    SLAM_L = StreamId("1201-1")
    SLAM_R = StreamId("1201-2")
    RGB = StreamId("214-1")
    ET = StreamId("211-1")
    
    def label(self) -> str:
        return {
            "ET": "camera-et",
            "RGB": "camera-rgb",
            "SLAM_L": "camera-slam-left",
            "SLAM_R": "camera-slam-right",
        }[self.name]

class CustomCalibration():
    def __init__(self, stream: Streams, device_calib):
        
        if stream != Streams.ET:
            self.original_calib = device_calib.get_camera_calib(stream.label())
            img_rgb_w, img_rgb_h = int(self.original_calib.get_image_size()[0]), int(self.original_calib.get_image_size()[1])

            self.pinhole_calib = calibration.get_linear_camera_calibration(
                                        #img_rgb_w, img_rgb_h,
                                        #calib_rgb_camera_original.get_focal_lengths()[0],
                                        512, 512, 200, #calib_rgb_camera_original.get_focal_lengths()[0],
                                        "pinhole",
                                        self.original_calib.get_transform_device_camera(),
                                        )
                        
            self.rotated_pinhole_calib = calibration.rotate_camera_calib_cw90deg(self.pinhole_calib)
            
    

class AriaProvider:
    
    
    def __init__(self, config_path):
        self.__config = tomllib.load(open(config_path, "rb"))
        self.__vrs_file = self.__config["aria_recordings"]["vrs"]
        #self.output_folder = self.__config["aria_recordings"]["output"]
        #self.gaze_output_folder = self.__config["aria_recordings"]["gaze_output"]
        
        self.__provider = data_provider.create_vrs_data_provider(self.__vrs_file)  
      
        self.t_first = self.__provider.get_first_time_ns(Streams.RGB.value, TimeDomain.DEVICE_TIME)
        self.t_last = self.__provider.get_last_time_ns(Streams.RGB.value, TimeDomain.DEVICE_TIME)
        
        self.calibration_device = self.__provider.get_device_calibration()
        self.customCalibrations: typing.Dict[Streams, CustomCalibration] = {}
        for s in Streams:
            self.customCalibrations[s] = CustomCalibration(s, self.calibration_device)  

    
    def get_time_range(self, time_step = 1e9):
        return range(self.t_first, self.t_last, int(time_step))
    
    def get_frame(self, stream: Streams, time_ns, rotated = True, undistorted = True):
        img = self.__provider.get_image_data_by_time_ns(
                stream.value, time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
            )[0].to_numpy_array()
        
        
        if undistorted:
            if rotated:
                img = np.rot90(img, -1).copy()
                img = calibration.distort_by_calibration(img, self.customCalibrations[stream].rotated_pinhole_calib, self.customCalibrations[stream].original_calib)
            else:
                img = calibration.distort_by_calibration(img, self.customCalibrations[stream].pinhole_calib, self.customCalibrations[stream].original_calib)
        elif rotated:
            img = np.rot90(img, -1).copy()
        
        return img


if __name__ == "__main__":
    print(Streams.ET)

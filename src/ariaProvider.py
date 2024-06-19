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
from projectaria_tools.core import data_provider, image
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
    


if __name__ == "__main__":
    print(Streams.ET.label())

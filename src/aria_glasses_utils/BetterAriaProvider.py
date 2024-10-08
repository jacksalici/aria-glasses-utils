#general libraries
import os
import shutil
import cv2
import numpy as np
import tomllib
import torch
from enum import Enum
import typing
import cv2

import aria.sdk as aria


#project aria libraries
from projectaria_tools.core import data_provider, image, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions, ImageDataRecord
from projectaria_tools.core.sensor_data import ImageDataRecord

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)



#project classes
from aria_glasses_utils.common import *


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
    
    def ariaCameraId(self):
        return {
            "ET": aria.CameraId.EyeTrack,
            "RGB": aria.CameraId.Rgb,
            "SLAM_L": aria.CameraId.Slam1,
            "SLAM_R": aria.CameraId.Slam2,
        }[self.name]
        
        
class StreamingClientObserver:
            def __init__(self):
                self.imgs = {}

            def on_image_received(self, image: np.array, record: ImageDataRecord):
                self.imgs[record.camera_id] = image

class CustomCalibration():
    def __init__(self, stream: Streams, device_calib, fix_size = False):
        
        if stream != Streams.ET:
            self.original_calib = device_calib.get_camera_calib(stream.label())
            
            if fix_size:
                self.img_w, self.img_h, self.img_f = 512, 512, 150
            else:    
                self.img_w, self.img_h, self.img_f = int(self.original_calib.get_image_size()[0]), int(self.original_calib.get_image_size()[1]), self.original_calib.get_focal_lengths()[0]

            self.pinhole_calib = calibration.get_linear_camera_calibration(
                                        self.img_w, self.img_h, self.img_f,
                                        "pinhole_"+stream.name,
                                        self.original_calib.get_transform_device_camera(),
                                        )
                        
            self.rotated_pinhole_calib = calibration.rotate_camera_calib_cw90deg(self.pinhole_calib)
            
class BetterAriaProvider:
    
    def __init__(self, live = False, vrs = None, profile_name = "profile18", streaming_interface = "usb", device_ip = None, cameras = "RGB-ET", verbose=True):
        self.live = live
        if not live:
            assert os.path.isfile(vrs), "VRS is mandatory streaming is not live."
      
            self.__provider = data_provider.create_vrs_data_provider(vrs)  
        
            self.t_first = self.__provider.get_first_time_ns(Streams.RGB.value, TimeDomain.DEVICE_TIME)
            self.t_last = self.__provider.get_last_time_ns(Streams.RGB.value, TimeDomain.DEVICE_TIME)
            
            self.calibration_device = self.__provider.get_device_calibration()
        
        elif live:
            if verbose:
                aria.set_log_level(aria.Level.Info)

            self.__device_client = aria.DeviceClient()
            client_config = aria.DeviceClientConfig()
            
            if device_ip:
                client_config.ip_v4_address = device_ip
                
            self.__device_client.set_client_config(client_config)
            
            self.__device = self.__device_client.connect()

            self.__streaming_manager = self.__device.streaming_manager
            self.__streaming_client = self.__streaming_manager.streaming_client

            streaming_config = aria.StreamingConfig()
            streaming_config.profile_name = profile_name

            if streaming_interface == "usb":
                print("Mode: USB")
                streaming_config.streaming_interface = aria.StreamingInterface.Usb
            
            streaming_config.security_options.use_ephemeral_certs = True
            self.__streaming_manager.streaming_config = streaming_config

            sensors_calib_json = self.__streaming_manager.sensors_calibration()
            self.calibration_device = device_calibration_from_json_string(sensors_calib_json)

            self.__streaming_manager.start_streaming()

            config = self.__streaming_client.subscription_config
                
            if cameras == "RGB-ET":
                config.subscriber_data_type = (aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack)
                config.message_queue_size[aria.StreamingDataType.Rgb] = 1
                config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
            elif cameras == "RGB":
                config.subscriber_data_type = aria.StreamingDataType.Rgb    
            
            self.__streaming_client.subscription_config = config

            self.observer = StreamingClientObserver()
            self.__streaming_client.set_streaming_client_observer(self.observer)
            self.__streaming_client.subscribe()
                
        self.customCalibrations: typing.Dict[Streams, CustomCalibration] = {}
        for s in Streams:
            #fix size of calibration only when live
            self.customCalibrations[s] = CustomCalibration(s, self.calibration_device, fix_size=live)          
    
    def unsubscribe(self):
        print("Stop listening to image data")
        self.__streaming_client.unsubscribe()
        self.__streaming_manager.stop_streaming()
        self.__device_client.disconnect(self.__device)
        
    def get_time_range(self, time_step = 1e9):
        assert not self.live, "How can you get the time range in a live streaming?"
        return range(self.t_first, self.t_last, int(time_step))
   

    def get_frame(self, stream: Streams, time_ns = None, rotated = True, undistorted = True, sync=True):
        if not self.live:
            assert time_ns, "Time must be specified."
            domain = TimeDomain.DEVICE_TIME
            mode = self.__provider.get_metadata().time_sync_mode.name
            if mode == 'TicSyncServer' or sync == False:
                domain = TimeDomain.DEVICE_TIME
            elif mode == 'TicSyncClient':
                domain = TimeDomain.TIC_SYNC
            else:
                raise NotImplementedError(f'Unsupported time-sync mode {mode}')
        
            img = self.__provider.get_image_data_by_time_ns(
                    stream.value, time_ns, domain, TimeQueryOptions.CLOSEST
                )[0].to_numpy_array()
                    
        else:
            if stream.ariaCameraId() in self.observer.imgs:
                img = self.observer.imgs[stream.ariaCameraId()]
                del self.observer.imgs[stream.ariaCameraId()]
            else:
                return None, False

        if stream is not Streams.ET:
            #not needed img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            if undistorted:
                if rotated:
                    img = np.rot90(img, k=3).copy()
                    img = calibration.distort_by_calibration(img, self.customCalibrations[stream].rotated_pinhole_calib, self.customCalibrations[stream].original_calib)
                else:
                    img = calibration.distort_by_calibration(img, self.customCalibrations[stream].pinhole_calib, self.customCalibrations[stream].original_calib)
            elif rotated:
                img = np.rot90(img, -1).copy()

        return img, True
    
    def get_calibration(self) -> tuple[typing.Dict[Streams, CustomCalibration], calibration.DeviceCalibration]:
        return self.customCalibrations, self.calibration_device
        
            

if __name__ == "__main__":
    config = tomllib.load(open("config.toml", "rb"))

    
    LIVE = True

    args = parse_args()

    provider = BetterAriaProvider(live=LIVE, vrs=config["aria_recordings"]["vrs"])
    #aria_streaming.init_cv2_windows("ET")

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ET", cv2.WINDOW_NORMAL)
    
    
    if LIVE:
        with ctrl_c_handler() as ctrl_c:
            while not (quit_keypress() or ctrl_c):
                img, success = provider.get_frame(Streams.RGB)
                if success:
                    cv2.imshow('RGB', img)
                
                img, success = provider.get_frame(Streams.ET)
                if success:
                    cv2.imshow('ET', img)
        
        provider.unsubscribe()

    else:
        for time in provider.get_time_range():
            img, _ = provider.get_frame(Streams.RGB, time_ns=time)
            img_et, _ = provider.get_frame(Streams.ET, time_ns=time)
            cv2.imshow('RGB', img)
            cv2.imshow('ET', img_et)
            
            
            if quit_keypress():
                break
            
        
            



            

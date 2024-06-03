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

from functools import reduce

import math
import cv2
from utils import *

class AriaStreaming:
    def __init__(self, args, log_level = aria.Level.Info, cameras = 'RGB'):
        self.args = args
        if args.update_iptables and sys.platform.startswith("linux"):
            update_iptables()
            
        aria.set_log_level(log_level)

        self.device_client = aria.DeviceClient()

        client_config = aria.DeviceClientConfig()
        if args.device_ip:
            client_config.ip_v4_address = args.device_ip
        self.device_client.set_client_config(client_config)

        
        self.device = self.device_client.connect()

        self.streaming_manager = self.device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = args.profile_name

        if args.streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config

        sensors_calib_json = self.streaming_manager.sensors_calibration()
        self.sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        self.rgb_calib = self.sensors_calib.get_camera_calib("camera-rgb")

        self.dst_calib = get_linear_camera_calibration(512, 512, 150, "camera-rgb")

        self.streaming_manager.start_streaming()

        config = self.streaming_client.subscription_config
        

            
        if cameras == "RGB-ET":
            config.subscriber_data_type = (aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack)
            config.message_queue_size[aria.StreamingDataType.Rgb] = 1
            config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
        elif cameras == "RGB":
            config.subscriber_data_type = aria.StreamingDataType.Rgb
        
        
        
        self.streaming_client.subscription_config = config

        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)
        self.streaming_client.subscribe()

    def init_cv2_windows(self, window):

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 512, 512)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(window, 50, 50)
    
    def unsubscribe(self):
        print("Stop listening to image data")
        self.streaming_client.unsubscribe()
        self.streaming_manager.stop_streaming()
        self.device_client.disconnect(self.device)

        

class StreamingClientObserver:
            def __init__(self):
                self.imgs = {}

            def on_image_received(self, image: np.array, record: ImageDataRecord):
                self.imgs[record.camera_id] = image
                    

def main():
    args = parse_args()

    aria_streaming = AriaStreaming(args, cameras = "RGB-ET")
    aria_streaming.init_cv2_windows("RGB")
    aria_streaming.init_cv2_windows("ET")

    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if aria.CameraId.Rgb in aria_streaming.observer.imgs:
                rgb_image = np.rot90(aria_streaming.observer.imgs[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                cv2.imshow('RGB', rgb_image)
                del aria_streaming.observer.imgs[aria.CameraId.Rgb]
                
            if aria.CameraId.EyeTrack in aria_streaming.observer.imgs:
                image = aria_streaming.observer.imgs[aria.CameraId.EyeTrack]
                cv2.imshow('ET', image)
                del aria_streaming.observer.imgs[aria.CameraId.EyeTrack]

                    


    aria_streaming.unsubscribe()

if __name__ == "__main__":
    main()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device_ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--correct_distorsion",
        help="Apply a correction to the distortion camera",
        default=True,
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
        
    model, classNames = init_yolo()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to the device
    device = device_client.connect()

    # 3. Retrieve the device streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # 4. Use a custom configuration for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    # Note: by default streaming uses Wifi
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # 5. Get sensors calibration
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")

    dst_calib = get_linear_camera_calibration(512, 512, 150, "camera-rgb")

    # 6. Start streaming
    streaming_manager.start_streaming()

    # 7. Configure subscription to listen to Aria's RGB stream.
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    streaming_client.subscription_config = config

    # 8. Create and attach the visualizer and start listening to streaming data
    class StreamingClientObserver:
        def __init__(self):
            self.img = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.img = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 9. Render the streaming data until we close the window
    window = "Aria Vision"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 512, 512)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window, 50, 50)


    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if observer.img is not None:
                img = cv2.cvtColor(observer.img, cv2.COLOR_BGR2RGB)
                
                if args.correct_distorsion:
                    # Apply the undistortion correction
                    img = distort_by_calibration(
                        img, dst_calib, rgb_calib
                    )
                
                results = model(img, stream=True, verbose=False)

                # coordinates
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
                        )  # convert to int values

                        # put box in cam
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        #print("Confidence --->", confidence)

                        # class name
                        cls = int(box.cls[0])
                        #print("Class name -->", classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(
                            img, classNames[cls], org, font, fontScale, color, thickness
                        )
                cv2.imshow(window, np.rot90(img, -1))

                

                observer.img = None

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)


if __name__ == "__main__":
    main()

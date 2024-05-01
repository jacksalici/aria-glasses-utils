

import argparse
import sys

import aria.sdk as aria

import cv2
import numpy as np

from common import ctrl_c_handler, quit_keypress, update_iptables

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

import signal
import subprocess
from contextlib import contextmanager

import cv2


def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)


@contextmanager
def ctrl_c_handler(signal_handler=None):
    class ctrl_c_state:
        def __init__(self):
            self._caught_ctrl_c = False

        def __bool__(self):
            return self._caught_ctrl_c

    state = ctrl_c_state()

    def _handler(sig, frame):
        state._caught_ctrl_c = True
        if signal_handler:
            signal_handler()

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)

    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")



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
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

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
            self.rgb_image = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.rgb_image = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 9. Render the streaming data until we close the window
    rgb_window = "Aria RGB"
    undistorted_window = "Undistorted RGB"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 512, 512)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(undistorted_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(undistorted_window, 512, 512)
    cv2.setWindowProperty(undistorted_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(undistorted_window, 600, 50)

    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if observer.rgb_image is not None:
                rgb_image = cv2.cvtColor(observer.rgb_image, cv2.COLOR_BGR2RGB)
                cv2.imshow(rgb_window, np.rot90(rgb_image, -1))

                # Apply the undistortion correction
                undistorted_rgb_image = distort_by_calibration(
                    rgb_image, dst_calib, rgb_calib
                )
                # Show the undistorted image
                cv2.imshow(undistorted_window, np.rot90(undistorted_rgb_image, -1))

                observer.rgb_image = None

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)


if __name__ == "__main__":
    main()

import projectaria_tools.core.mps as mps
# Example query: find the nearest eye gaze data outputs in relation to a specific timestamp
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze
)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from time import sleep
def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")


eye_gaze_path = "/Users/jacksalici/Desktop/AriaRecording/test6/mps_e3adbe21-c424-4d3e-b16a-d4af25bd5d4f_vrs/eye_gaze/general_eye_gaze.csv"
vrs_file = "/Users/jacksalici/Desktop/AriaRecording/test6/e3adbe21-c424-4d3e-b16a-d4af25bd5d4f.vrs"
gaze_cpfs = mps.read_eyegaze(eye_gaze_path)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)

provider = data_provider.create_vrs_data_provider(vrs_file)

rgb_stream_id = StreamId("214-1")
rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)

t_first = provider.get_first_time_ns(rgb_stream_id, TimeDomain.DEVICE_TIME)
t_last = provider.get_last_time_ns(rgb_stream_id, TimeDomain.DEVICE_TIME)



calib_device = provider.get_device_calibration()
calib_rgb_camera_original = calib_device.get_camera_calib(rgb_stream_label)
img_w, img_h = int(calib_rgb_camera_original.get_image_size()[0]), int(calib_rgb_camera_original.get_image_size()[1])
calib_rgb_camera_linear = calibration.get_linear_camera_calibration(
                img_w, img_h,
                calib_rgb_camera_original.get_focal_lengths()[0],
                "pinhole",
                calib_rgb_camera_original.get_transform_device_camera(),
                )
calib_rgb_camera = calibration.rotate_camera_calib_cw90deg(calib_rgb_camera_linear)

for time in range(t_first, t_last, 1000000000):
    gaze_cpf = get_nearest_eye_gaze(gaze_cpfs, time)

   
    gaze_center_in_cpf = mps.get_eyegaze_point_at_depth(gaze_cpf.yaw, gaze_cpf.pitch, gaze_cpf.depth or 1.0)
    transform_cpf_sensor = calib_device.get_transform_cpf_sensor(rgb_stream_label)
    gaze_center_in_camera = transform_cpf_sensor.inverse() @ gaze_center_in_cpf
    gaze_center_in_pixels = calib_rgb_camera.project(gaze_center_in_camera).astype(int)
    
    print("GAZE CENTER IN CPF:", gaze_center_in_cpf)
    print("GAZE CENTER IN PIXEL:", gaze_center_in_pixels)

    img = provider.get_image_data_by_time_ns(rgb_stream_id, time, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)[0].to_numpy_array()

    img = np.rot90(img, -1).copy()
    img = calibration.distort_by_calibration(img, calib_rgb_camera, calib_rgb_camera_original)

 
    cv2.circle(img, [img_w-int(gaze_center_in_pixels[1]), int(gaze_center_in_pixels[0])] , 5, (255, 0, 0), 2)
    

    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", img)
    sleep(0.3)
    if quit_keypress():
        break
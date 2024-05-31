import projectaria_tools.core.mps as mps
# Example query: find the nearest eye gaze data outputs in relation to a specific timestamp
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze
)
import matplotlib.pyplot as plt
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
t_first = provider.get_first_time_ns(rgb_stream_id, TimeDomain.DEVICE_TIME)
t_last = provider.get_last_time_ns(rgb_stream_id, TimeDomain.DEVICE_TIME)
rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
device_calibration = provider.get_device_calibration()
rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)


for time in range(t_first, t_last, 1000000000):
    gaze_cpf = get_nearest_eye_gaze(gaze_cpfs, time)
    depth_m = gaze_cpf.depth or 1.0
    gaze_point_cpf = mps.get_eyegaze_point_at_depth(gaze_cpf.yaw, gaze_cpf.pitch, depth_m)

    print(gaze_point_cpf)
   

    gaze_projection = get_gaze_vector_reprojection(
                                gaze_cpf,
                                rgb_stream_label,
                                device_calibration,
                                rgb_camera_calibration,
                                depth_m,
                            )
    image_data = provider.get_image_data_by_time_ns(rgb_stream_id, time, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)

       
    raw_image = image_data[0].to_numpy_array()


    # Convert the image from BGR to RGB format
    cv2.circle(raw_image, [int(gaze_projection[0]), int(gaze_projection[1])] , 5, (255, 0, 0), 2)


        
        
    print(gaze_projection)

    
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", raw_image)
    sleep(1)
    if quit_keypress():
        break
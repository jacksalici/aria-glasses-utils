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

def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")


eye_gaze_path = "/Users/jacksalici/Desktop/AriaRecording/test6/mps_e3adbe21-c424-4d3e-b16a-d4af25bd5d4f_vrs/eye_gaze/general_eye_gaze.csv"
vrs_file = "/Users/jacksalici/Desktop/AriaRecording/test6/e3adbe21-c424-4d3e-b16a-d4af25bd5d4f.vrs"
gaze_cpfs = mps.read_eyegaze(eye_gaze_path)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)

for index, gaze_cpf in enumerate(gaze_cpfs):
    depth_m = gaze_cpf.depth or 1.0
    gaze_point_cpf = mps.get_eyegaze_point_at_depth(gaze_cpf.yaw, gaze_cpf.pitch, depth_m)


    query_timestamp_ns = int(gaze_cpf.tracking_timestamp.total_seconds() * 1e9)

    eye_gaze_info = get_nearest_eye_gaze(gaze_cpfs, query_timestamp_ns)

    if eye_gaze_info:
        # Re-project the eye gaze point onto the RGB camera data
        
        provider = data_provider.create_vrs_data_provider(vrs_file)

        rgb_stream_id = StreamId("214-1")
        rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
        device_calibration = provider.get_device_calibration()
        rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

        gaze_projection = get_gaze_vector_reprojection(
                                eye_gaze_info,
                                rgb_stream_label,
                                device_calibration,
                                rgb_camera_calibration,
                                depth_m,
                            )
        num_rgb_frames = provider.get_num_data(rgb_stream_id)
        imagebyindex = provider.get_image_data_by_index(rgb_stream_id, index)
        print(imagebyindex)
        raw_image = imagebyindex[0].to_numpy_array()


        # Convert the image from BGR to RGB format
        cv2.circle(raw_image, [int(gaze_projection[0]), int(gaze_projection[1])] , 5, (255, 0, 0), 2)


        
        
        print(gaze_projection)

    
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", raw_image)
        
        if quit_keypress():
            break
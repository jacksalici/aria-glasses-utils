from projectaria_tools.core import data_provider, image, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from utils import *


from eyeGaze import EyeGaze
from gazeInference import GazeInference
import cv2
import numpy as np

import os




def main():
    import tomllib
    config = tomllib.load(open("config.toml", "rb"))
    vrs_file = config["aria_recordings"]["vrs"]
 
    
    stream_label_rgb = "camera-rgb"
    stream_label_et = "camera-et"
    stream_label_slam_l = "camera-slam-left"
    stream_label_slam_r = "camera-slam-right"
    provider = data_provider.create_vrs_data_provider(vrs_file)

    stream_id_rgb = provider.get_stream_id_from_label(stream_label_rgb)
    stream_id_et = provider.get_stream_id_from_label(stream_label_et)
    stream_id_slam_l = provider.get_stream_id_from_label(stream_label_slam_l)
    stream_id_slam_r = provider.get_stream_id_from_label(stream_label_slam_r)
    
    
    
    t_first = provider.get_first_time_ns(stream_id_rgb, TimeDomain.DEVICE_TIME)
    t_last = provider.get_last_time_ns(stream_id_rgb, TimeDomain.DEVICE_TIME)

    calib_device = provider.get_device_calibration()
    
    # CALIBRATION RGB
    
    calib_rgb_camera_original = calib_device.get_camera_calib(stream_label_rgb)
    img_rgb_w, img_rgb_h = int(calib_rgb_camera_original.get_image_size()[0]), int(calib_rgb_camera_original.get_image_size()[1])

    
    calib_rgb_camera = calibration.get_linear_camera_calibration(
                                img_rgb_w, img_rgb_h,
                                calib_rgb_camera_original.get_focal_lengths()[0],
                                "pinhole",
                                calib_rgb_camera_original.get_transform_device_camera(),
                                )
                
    calib_rgb_camera = calibration.rotate_camera_calib_cw90deg(calib_rgb_camera)
    
    # CALIBRATION SLAM LEFT
        
    calib_slam_l_camera_original = calib_device.get_camera_calib(stream_label_slam_l)
    img_slam_l_w, img_slam_l_h = int(calib_slam_l_camera_original.get_image_size()[0]), int(calib_slam_l_camera_original.get_image_size()[1])

    calib_slam_l_camera = calibration.get_linear_camera_calibration(
                                img_slam_l_w, img_slam_l_h,
                                calib_slam_l_camera_original.get_focal_lengths()[0],
                                "pinhole",
                                calib_slam_l_camera_original.get_transform_device_camera(),
                                )
                
    calib_slam_l_camera = calibration.rotate_camera_calib_cw90deg(calib_slam_l_camera)
                    
  
    extrinsic_cam_rgb = calib_device.get_transform_cpf_sensor(
        stream_label_rgb
    ).to_matrix()
    
    extrinsic_cam_slam_l = calib_device.get_transform_cpf_sensor(
        stream_label_slam_l
    ).to_matrix()
    
    

    imgs = []
    imgs_slam_l = []
    
    for time in range(t_first, t_last, int(1000_000_000/config["depth_estimator"]["frame_per_seconds"])):
        print(f"INFO: Checking frame at time {time}")

        
        img_rgb = provider.get_image_data_by_time_ns(
            stream_id_rgb, time, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )[0].to_numpy_array()
        
        img_slam_l = provider.get_image_data_by_time_ns(
            stream_id_slam_l, time, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )[0].to_numpy_array()
        
        img_rgb = calibration.distort_by_calibration(np.rot90(img_rgb, -1).copy(), calib_rgb_camera, calib_rgb_camera_original)
        
        

        #imgs.append(img)
        cv2.imshow("RGB", cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        
        img_slam_l = calibration.distort_by_calibration(np.rot90(img_slam_l, -1).copy(), calib_slam_l_camera, calib_slam_l_camera_original)
        cv2.imshow("SLAM L", img_slam_l)
        
        
        
        cv2.waitKey()
        
        # cv2.circle(img, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 2)
        # sleep(0.3)

    import json, torch

    
if __name__ == "__main__":
    main()

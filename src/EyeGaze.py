import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_nearest_eye_gaze
import matplotlib.pyplot as plt
import numpy as np
import cv2
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from time import sleep

from utils import *

class EyeGaze:
    def __init__(self, live: bool, correct_distorsion: bool = False, rotate_image: bool = True, vrs_file = None) -> None:
        self.live, self.correct_distortion, self.rotate_image = live, correct_distorsion, rotate_image
        self.stream_ids = {
            "slam-l": StreamId("1201-1"),
            "slam-r": StreamId("1201-2"),
            "rgb": StreamId("214-1"),
            "et": StreamId("211-1"),
        }
             
        if not live:
            assert vrs_file, "VRS file is mandatory in not live streaming."
            
            self.provider = data_provider.create_vrs_data_provider(vrs_file)
            self.stream_labels = {k: self.provider.get_label_from_stream_id(v) for k, v in self.stream_ids.items()}
            self.t_first = self.provider.get_first_time_ns(self.stream_ids['rgb'], TimeDomain.DEVICE_TIME)
            self.t_last = self.provider.get_last_time_ns(self.stream_ids['rgb'], TimeDomain.DEVICE_TIME)

            self.calib_device = self.provider.get_device_calibration()
            
            
            self.calib_rgb_camera_original = self.calib_device.get_camera_calib(self.stream_labels['rgb'])
            self.img_w, self.img_h = int(self.calib_rgb_camera_original.get_image_size()[0]), int(self.calib_rgb_camera_original.get_image_size()[1])
            
            if correct_distorsion:
                self.calib_rgb_camera = calibration.get_linear_camera_calibration(
                                self.img_w, self.img_h,
                                self.calib_rgb_camera_original.get_focal_lengths()[0],
                                "pinhole",
                                self.calib_rgb_camera_original.get_transform_device_camera(),
                                )
                
                if rotate_image:
                    self.calib_rgb_camera = calibration.rotate_camera_calib_cw90deg(self.calib_rgb_camera)
                    
            else:
                self.calib_rgb_camera = self.calib_rgb_camera_original
                
                if rotate_image:
                    print("WARNING: Calibration cannot be rotated without undistortion.")
               
    
    def get_gaze_center(self, gaze_cpf):
        gaze_center_in_cpf = mps.get_eyegaze_point_at_depth(gaze_cpf.yaw, gaze_cpf.pitch, gaze_cpf.depth or 1.0)
        transform_cpf_sensor = self.calib_device.get_transform_cpf_sensor(self.stream_labels['rgb'])
        gaze_center_in_camera = transform_cpf_sensor.inverse() @ gaze_center_in_cpf
        gaze_center_in_pixels = self.calib_rgb_camera.project(gaze_center_in_camera).astype(int)
        return gaze_center_in_cpf, gaze_center_in_pixels
    
    def rotate_pixel_cw90(self, gaze_center_in_pixels):
        return [self.img_w-gaze_center_in_pixels[1], gaze_center_in_pixels[0]]
    
    def get_rgb_image(self, time_ns = None, index = None):
        assert not (time_ns == None and index == None), "Time or Index must be specified"
        
        if time_ns:
            img = self.provider.get_image_data_by_time_ns(self.stream_ids['rgb'], time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)[0].to_numpy_array()

        if self.rotate_image:
            img = np.rot90(img, -1).copy()
        
        if self.correct_distortion:
            img = calibration.distort_by_calibration(img, self.calib_rgb_camera, self.calib_rgb_camera_original)
        
        return img
    
    def get_time_range(self, time_step = 1000000000):
        return range(self.t_first, self.t_last, time_step)
    
    
    
    
    
def main():
    import tomllib
    config = tomllib.load(open('config.toml', 'rb'))

    eye_gaze_path = config['aria_recordings'][0]['general_eye_gaze']
    vrs_file = config['aria_recordings'][0]['vrs']
    gaze_cpfs = mps.read_eyegaze(eye_gaze_path)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)

    eye_gaze = EyeGaze(live = False, correct_distorsion=False, rotate_image=True, vrs_file=vrs_file)

    for time in eye_gaze.get_time_range():
        gaze_cpf = get_nearest_eye_gaze(gaze_cpfs, time)

        gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center(gaze_cpf)

        img = eye_gaze.get_rgb_image(time_ns=time)
        
        cv2.circle(img, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        sleep(0.3)
        
        if quit_keypress():
            break
        
if __name__ == "__main__":
    main()
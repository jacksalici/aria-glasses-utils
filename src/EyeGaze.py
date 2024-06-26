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

from gazeInference import GazeInference

class EyeGaze:
    def __init__(self, live: bool, correct_distorsion: bool = False, rotate_image: bool = True, vrs_file = None, calib_live = None) -> None:
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
            
        if live:
            assert calib_live, "Device calibration must provided when live is done."
            self.stream_labels = {'rgb': 'camera-rgb', 'et': 'camera-et'}
            self.calib_device = calib_live
            
        self.calib_rgb_camera_original = self.calib_device.get_camera_calib(self.stream_labels['rgb'])
        self.img_w, self.img_h = int(self.calib_rgb_camera_original.get_image_size()[0]), int(self.calib_rgb_camera_original.get_image_size()[1])

            
        if correct_distorsion:
            self.calib_rgb_camera_pinhole = calibration.get_linear_camera_calibration(
                                self.img_w, self.img_h,
                                # 1080, 1080,
                                self.calib_rgb_camera_original.get_focal_lengths()[0],
                                "pinhole",
                                self.calib_rgb_camera_original.get_transform_device_camera(),
                                )
                
            if rotate_image:
                self.calib_rgb_camera_pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(self.calib_rgb_camera_pinhole)
                    
        
    def get_gaze_center(self, gaze_cpf, legacy_model=True):
        if legacy_model:
            return self.get_gaze_center_raw(gaze_cpf.yaw, gaze_cpf.pitch, gaze_cpf.depth or 1.0)
        else:
            depth, combined_yaw, combined_pitch = (
                mps.compute_depth_and_combined_gaze_direction(
                   gaze_cpf.vergence.left_yaw, gaze_cpf.vergence.right_yaw, gaze_cpf.pitch
                )
            )
            return self.get_gaze_center_raw(combined_yaw, combined_pitch, depth)
    
    def get_gaze_center_raw (self, yaw, pitch, depth = 1.0):
        """Get the gaze center both in cpf and at depth

        Args:
            yaw: mps output
            pitch: mps output
            depth (float, optional). Defaults to 1.0.

        Returns:
            tuple: gaze_center_in_cpf, gaze_center_in_pixels
        """
        gaze_center_in_cpf = mps.get_eyegaze_point_at_depth(yaw, pitch, depth)
        transform_cpf_sensor = self.calib_device.get_transform_cpf_sensor(self.stream_labels['rgb'])
        gaze_center_in_camera = transform_cpf_sensor.inverse() @ gaze_center_in_cpf
        
        if self.correct_distortion:
            gaze_center_in_pixels = self.calib_rgb_camera_pinhole.project(gaze_center_in_camera).astype(int)
        else:
            gaze_center_in_pixels = self.calib_rgb_camera_original.project(gaze_center_in_camera).astype(int)
            
        if self.rotate_image:
            gaze_center_in_pixels = self.rotate_pixel_cw90(gaze_center_in_pixels)
        
        return gaze_center_in_cpf, gaze_center_in_pixels
        
    
    def rotate_pixel_cw90(self, gaze_center_in_pixels, scale = 1.0):
        return [int(self.img_w*scale) - gaze_center_in_pixels[1], gaze_center_in_pixels[0]]
    
    def get_rgb_image(self, time_ns):
        
        img = self.provider.get_image_data_by_time_ns(self.stream_ids['rgb'], time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)[0].to_numpy_array()

        if self.correct_distortion:
            img = calibration.distort_by_calibration(img, self.calib_rgb_camera_pinhole, self.calib_rgb_camera_original)

        if self.rotate_image:
            img = np.rot90(img, -1).copy()
        
        return img
    
    def getCameraCalib(self):
        if self.correct_distortion:
            if self.rotate_image:
                return self.calib_rgb_camera_pinhole_cw90
            return self.calib_rgb_camera_pinhole
        return self.calib_rgb_camera_original

        
        

    def get_et_image(self, time_ns = None):
        
        img = self.provider.get_image_data_by_time_ns(self.stream_ids['et'], time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)[0].to_numpy_array()

        
        return img
    
    def get_time_range(self, time_step = 100000000):
        return range(self.t_first, self.t_last, time_step)
    
    
    
    
    
def main():
    import tomllib
    config = tomllib.load(open('config.toml', 'rb'))

    eye_gaze_path = config['aria_recordings']['general_eye_gaze']
    vrs_file = config['aria_recordings']['vrs']
    gaze_cpfs = mps.read_eyegaze(eye_gaze_path)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    
    gaze_inf = GazeInference()

    eye_gaze = EyeGaze(live = False, correct_distorsion=True, rotate_image=True, vrs_file=vrs_file)

    for time in eye_gaze.get_time_range():
        gaze_cpf = get_nearest_eye_gaze(gaze_cpfs, time)
        if(gaze_cpf is None):
            continue
        gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center(gaze_cpf)

        img = eye_gaze.get_rgb_image(time_ns=time)
        img_et = eye_gaze.get_et_image(time_ns=time)
        
        yaw, pitch = gaze_inf.predict(gaze_inf.a2t(img_et)) 
        gaze_center_in_cpf2, gaze_center_in_pixels2 = eye_gaze.get_gaze_center_raw(yaw, pitch, 0.5)

        
        cv2.circle(img, gaze_center_in_pixels , 5, (255, 0, 0), 2)
        cv2.circle(img, gaze_center_in_pixels2 , 5, (255, 255, 0), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        
        if quit_keypress():
            break
            
        
if __name__ == "__main__":
    main()
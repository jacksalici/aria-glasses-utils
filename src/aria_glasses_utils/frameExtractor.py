from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from .utils import *


from .BetterEyeGaze import BetterEyeGaze
from .BetterAriaProvider import *


import cv2
import numpy as np

import os




def confidence(img1, img2):

    def process(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(img_gray, (43, 43), 21)

    res = cv2.matchTemplate(process(img1), process(img2), cv2.TM_CCOEFF_NORMED)
    return res.max()


def blurryness(img):
    return -cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main():
    import tomllib

    config = tomllib.load(open("config.toml", "rb"))
    
    provider = BetterAriaProvider("config.toml")

    output_folder = config["aria_recordings"]["output"]
    gaze_output_folder = config["aria_recordings"]["gaze_output"]
    
    import shutil
    shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)
    
    shutil.rmtree(gaze_output_folder, ignore_errors=True)
    os.mkdir(gaze_output_folder)
    
    eye_gaze = BetterEyeGaze(False, correct_distorsion=True, vrs_file=config["aria_recordings"]["vrs"])
    
    imgs = []
    imgs_et = []
    for time in provider.get_time_range():
        print(f"INFO: Checking frame at time {time}")
        frame = {}
        
        frame['rgb'] = provider.get_frame(Streams.RGB, time_ns=time)
        img_et = provider.get_frame(Streams.ET, time, False, False)
        
        frame['slam_l'] = provider.get_frame(Streams.SLAM_L, time)
        frame['slam_r'] = provider.get_frame(Streams.SLAM_R, time)
        
        


        if (len(imgs) > 0 and confidence(frame["rgb"], imgs[-1]["rgb"]) < 0.7) or len(imgs) == 0:
            imgs.append(frame)
            imgs_et.append(img_et)
            
            print(f"INFO: Frame added to the list.")
        else:
            if blurryness(frame["rgb"]) < blurryness(imgs[-1]["rgb"]):
                imgs[-1] = frame
                print(
                    f"INFO: Frame substituted to the last in the list for better sharpness."
                )

        # cv2.circle(img, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 2)
        # sleep(0.3)

    import torch
    
    extrinsic_cam_rgb = provider.calibration_device.get_transform_cpf_sensor(
        Streams.RGB.label()
    ).to_matrix()

    for index, frame in enumerate(imgs):
        
        for name,img in frame.items():
            cv2.imwrite(os.path.join(output_folder, f"img{index}{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        yaw, pitch = eye_gaze.predict(torch.tensor(imgs_et[index], device="cpu"))
        gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center_raw(
            yaw, pitch
        ) 

        np.savez(
                os.path.join(gaze_output_folder, f"img{index}.npz"),
                gaze_yaw_pitch=np.array([yaw, pitch]),
                gaze_center_in_cpf=gaze_center_in_cpf,
                gaze_center_in_rgb_pixels=gaze_center_in_pixels,
                gaze_center_in_rgb_frame=(
                    np.linalg.inv(extrinsic_cam_rgb)
                    @ np.append(gaze_center_in_cpf, [1])
                )[:3],
                rbg_camera_extrinsic=extrinsic_cam_rgb,
                rbg_camera_intrinsic=eye_gaze.getCameraCalib().projection_params(),
            )
        print(f"INFO: File {index} saved.")


if __name__ == "__main__":
    main()

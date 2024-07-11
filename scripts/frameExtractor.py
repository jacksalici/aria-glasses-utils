from aria_glasses_utils.common import *
from aria_glasses_utils.BetterEyeGaze import BetterEyeGaze
from aria_glasses_utils.BetterAriaProvider import *

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
    
    provider = BetterAriaProvider(vrs =  config["aria_recordings"]["vrs"])

    output_folder = config["aria_recordings"]["output"]
    gaze_output_folder = config["aria_recordings"]["gaze_output"]
    
    import shutil
    shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)
    
    shutil.rmtree(gaze_output_folder, ignore_errors=True)
    os.mkdir(gaze_output_folder)
    
    eye_gaze = BetterEyeGaze(*provider.get_calibration())
    
    imgs = []
    imgs_et = []
    for time in provider.get_time_range(1000000000):
        print(f"INFO: Checking frame at time {time}")
        frame = {}
        
        frame['rgb'], _ = provider.get_frame(Streams.RGB, time_ns=time)
        img_et, _ = provider.get_frame(Streams.ET, time, False, False)
        
        #frame['slam_l'], _ = provider.get_frame(Streams.SLAM_L, time)
        #frame['slam_r'], _ = provider.get_frame(Streams.SLAM_R, time)
        
        ALL_IMAGES = True
        if (len(imgs) > 0 and confidence(frame["rgb"], imgs[-1]["rgb"]) < 0.7) or len(imgs) == 0 or ALL_IMAGES:
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
    
    rbg2cpf_camera_extrinsic = provider.calibration_device.get_transform_cpf_sensor(
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
                    np.linalg.inv(rbg2cpf_camera_extrinsic)
                    @ np.append(gaze_center_in_cpf, [1])
                )[:3],
                rbg2cpf_camera_extrinsic=rbg2cpf_camera_extrinsic,
                rbg_camera_intrinsic=eye_gaze.getCameraCalib().projection_params(),
            )
        print(f"INFO: File {index} saved.")


if __name__ == "__main__":
    main()

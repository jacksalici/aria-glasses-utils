from aria_glasses_utils.common import *
from aria_glasses_utils.BetterEyeGaze import BetterEyeGaze
from aria_glasses_utils.BetterAriaProvider import *

import cv2
import numpy as np
from pathlib import Path
import torch
import os

def confidence(img1, img2):

    def process(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(img_gray, (43, 43), 21)

    res = cv2.matchTemplate(process(img1), process(img2), cv2.TM_CCOEFF_NORMED)
    return res.max()

def blurryness(img):
    return -cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def exportFrames(input_vrs_path, imgs_output_dir, gaze_output_folder = None, export_gaze_info = False, export_time_step = 1_000_000_000, export_slam_camera_frames = True, min_confidence = 0.7, show_preview = False):
    provider = BetterAriaProvider(vrs=input_vrs_path)
    Path(imgs_output_dir).mkdir( parents=True, exist_ok=True )
    imgs = []

    if export_gaze_info:
        assert gaze_output_folder, "gaze output folder must specified."
        eye_gaze = BetterEyeGaze(*provider.get_calibration())
        Path(gaze_output_folder).mkdir( parents=True, exist_ok=True )
            
        imgs_et = []
    
    if show_preview:
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imshow('PREVIEW', image)

    
    for time in provider.get_time_range(export_time_step):
        print(f"INFO: Checking frame at time {time}")
        frame = {}
        
        frame['rgb'], _ = provider.get_frame(Streams.RGB, time_ns=time)
        frame['rgb'] = cv2.cvtColor(frame["rgb"], cv2.COLOR_BGR2RGB)
        
        if export_gaze_info:
            img_et, _ = provider.get_frame(Streams.ET, time, False, False)
        
        if export_slam_camera_frames:
            frame['slam_l'], _ = provider.get_frame(Streams.SLAM_L, time)
            frame['slam_r'], _ = provider.get_frame(Streams.SLAM_R, time)
        
        
        if (len(imgs) > 0 and confidence(frame["rgb"], imgs[-1]["rgb"]) < min_confidence) or len(imgs) == 0:
            imgs.append(frame)
            
            if export_gaze_info:
                imgs_et.append(img_et)
            
            print(f"INFO: Frame added to the list.")
        else:
            if blurryness(frame["rgb"]) < blurryness(imgs[-1]["rgb"]):
                imgs[-1] = frame
                print(
                    f"INFO: Frame substituted to the last in the list for better sharpness."
                )

    rbg2cpf_camera_extrinsic = provider.calibration_device.get_transform_cpf_sensor(
        Streams.RGB.label()
    ).to_matrix()

    for index, frame in enumerate(imgs):
        
        for name,img in frame.items():
            cv2.imwrite(os.path.join(imgs_output_dir, f"img{index}{name}.jpg"), img)
        
        if export_gaze_info:
            yaw, pitch = eye_gaze.predict(torch.tensor(imgs_et[index], device="cpu"))
            gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center_raw(
                yaw, pitch
            ) 
            
            if show_preview:
                cv2.circle(frame["rgb"], gaze_center_in_pixels , 5, (255, 0, 0), 2)
                cv2.imshow('PREVIEW', frame["rgb"])
                key = cv2.waitKey(0)
                if key == ord('q'):
                    show_preview = False
                
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


def main():
    import tomllib

    config = tomllib.load(open("config.toml", "rb"))
    input_vrs_path = config["aria_recordings"]["vrs"]
    output_folder = config["aria_recordings"]["output"]
    gaze_output_folder =  config["aria_recordings"]["gaze_output"]
    
    exportFrames(input_vrs_path, output_folder, gaze_output_folder, True, show_preview=True)

if __name__ == "__main__":
    main()

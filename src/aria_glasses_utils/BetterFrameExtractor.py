from aria_glasses_utils.common import *
from aria_glasses_utils.BetterEyeGaze import BetterEyeGaze
from aria_glasses_utils.BetterAriaProvider import *

import cv2
import numpy as np
from pathlib import Path
import torch
import os

def similarity(img1, img2):

    def process(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(img_gray, (43, 43), 21)

    res = cv2.matchTemplate(process(img1), process(img2), cv2.TM_CCOEFF_NORMED)
    return res.max()

def blurriness(img):
    return -cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def exportFrames(input_vrs_path, imgs_output_dir, gaze_output_folder = None, export_gaze_info = False, export_time_step = 1_000_000_000, export_slam_camera_frames = False, max_similarity = 0.7, show_preview = False, range_limits_ns = None, filename_prefix="", filename_w_timestamp = True):
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

    if range_limits_ns:
        time_range = range(range_limits_ns[0], range_limits_ns[1], export_time_step)
    else:
        time_range = provider.get_time_range(export_time_step)
        
    for time in time_range:
        print(f"INFO: Checking frame at time {time}")
        frame = {}
        frame['timestamp'] = str(time)
        frame['rgb'], _ = provider.get_frame(Streams.RGB, time_ns=time)
        frame['rgb'] = cv2.cvtColor(frame["rgb"], cv2.COLOR_BGR2RGB)
        
        if export_gaze_info:
            img_et, _ = provider.get_frame(Streams.ET, time, False, False)
        
        if export_slam_camera_frames:
            frame['slam_l'], _ = provider.get_frame(Streams.SLAM_L, time)
            frame['slam_r'], _ = provider.get_frame(Streams.SLAM_R, time)
        
        if show_preview:
            image = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.imshow('PREVIEW', image)
        
        if (len(imgs) > 0 and similarity(frame["rgb"], imgs[-1]["rgb"]) < max_similarity) or len(imgs) == 0:
            imgs.append(frame)
            
            if export_gaze_info:
                imgs_et.append(img_et)
            
            print(f"INFO: Frame added to the list.")
        else:
            if blurriness(frame["rgb"]) < blurriness(imgs[-1]["rgb"]):
                imgs[-1] = frame
                if export_gaze_info:
                    imgs_et[-1] = img_et
                print(
                    f"INFO: Frame substituted to the last in the list for better sharpness."
                )
            else:
                print("WARNING: Frame not addded")

    rbg2cpf_camera_extrinsic = provider.calibration_device.get_transform_cpf_sensor(
        Streams.RGB.label()
    ).to_matrix()

    for index, frame in enumerate(imgs):
        
        file_name = filename_prefix + frame['timestamp'] if filename_w_timestamp else str(index)
        
        cv2.imwrite(os.path.join(imgs_output_dir, f"{file_name}.jpg"), frame['rgb'])

        if export_slam_camera_frames:
            cv2.imwrite(os.path.join(imgs_output_dir, f"{file_name}_slam_l.jpg"), frame['slam_l'])
            cv2.imwrite(os.path.join(imgs_output_dir, f"{file_name}_slam_r.jpg"), frame['slam-r'])
            
        
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
                    os.path.join(gaze_output_folder, f"{file_name}.npz"),
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
    gaze_output = "N"
    export_slam = "N"
    
    input_vrs_path = input(f"Input VRS path: [{input_vrs_path}]:") or input_vrs_path
    output_folder = input(f"Output saving path [{output_folder}]:") or output_folder
    export_slam = input(f"Export also slam frame? [y/N]") or "N"
    
    gaze_output = input(f"Gaze output? [y/N]") or "N"
    
    if gaze_output == 'y':
         gaze_output_folder = input(f"Gaze info output folder [{gaze_output_folder}]:") or gaze_output_folder
    
    
    exportFrames(input_vrs_path, output_folder, gaze_output_folder, gaze_output == 'y', export_slam_camera_frames = export_slam == 'y', show_preview=True)

if __name__ == "__main__":
    main()

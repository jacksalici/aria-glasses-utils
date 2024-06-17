from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from utils import *


from eyeGaze import EyeGaze
from gazeInference import GazeInference
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

    vrs_file = config["aria_recordings"]["vrs"]
    output_folder = config["aria_recordings"]["output"]

    import shutil

    shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)
    stream_label_rgb = "camera-rgb"
    stream_label_et = "camera-et"
    provider = data_provider.create_vrs_data_provider(vrs_file)

    stream_id_rgb = provider.get_stream_id_from_label(stream_label_rgb)
    stream_id_et = provider.get_stream_id_from_label(stream_label_et)
    t_first = provider.get_first_time_ns(stream_id_rgb, TimeDomain.DEVICE_TIME)
    t_last = provider.get_last_time_ns(stream_id_rgb, TimeDomain.DEVICE_TIME)

    calib_device = provider.get_device_calibration()

    eye_gaze = EyeGaze(
        live=False, correct_distorsion=True, rotate_image=True, vrs_file=vrs_file
    )
    eye_gaze_inf = GazeInference()

    rbg_camera_extrinsic = calib_device.get_transform_cpf_sensor(
        stream_label_rgb
    ).to_matrix()

    imgs = []
    imgs_et = []
    for time in range(t_first, t_last, 1000_000_000):
        print(f"INFO: Checking frame at time {time}")

        img = eye_gaze.get_rgb_image(time_ns=time)
        img_et = provider.get_image_data_by_time_ns(
            stream_id_et, time, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )[0].to_numpy_array()

        if len(imgs) == 0:
            imgs.append(img)
            imgs_et.append(img_et)
            print(f"INFO: Frame added to the list.")

        if len(imgs) > 0 and confidence(img, imgs[-1]) < 0.7:
            imgs.append(img)
            imgs_et.append(img_et)
            print(f"INFO: Frame added to the list.")
        else:
            if blurryness(img) < blurryness(imgs[-1]):
                imgs[-1] = img
                print(
                    f"INFO: Frame substituted to the last in the list for better sharpness."
                )

        # cv2.circle(img, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 2)
        # sleep(0.3)

    import json, torch

    for index, img in enumerate(imgs):
        cv2.imwrite(os.path.join(output_folder, f"img{index}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        yaw, pitch = eye_gaze_inf.predict(torch.tensor(imgs_et[index], device="cpu"))
        gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center_raw(
            yaw, pitch
        ) 

        np.savez(
                os.path.join(output_folder, f"img{index}.npz"),
                gaze_center_in_cpf=gaze_center_in_cpf,
                gaze_center_in_rgb_pixels=gaze_center_in_pixels,
                gaze_center_in_rgb_frame=(
                    np.linalg.inv(rbg_camera_extrinsic)
                    @ np.append(gaze_center_in_cpf, [1])
                )[:3],
                rbg_camera_extrinsic=rbg_camera_extrinsic,
                rbg_camera_intrinsic=eye_gaze.calib_rgb_camera.projection_params(),
            )
        print(f"INFO: File {index} saved.")


if __name__ == "__main__":
    main()

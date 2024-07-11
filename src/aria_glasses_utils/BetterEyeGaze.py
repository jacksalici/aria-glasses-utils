import sys
import tomllib
import matplotlib.pyplot as plt
import cv2
import os, time
import torch
import numpy as np
from pathlib import Path

import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_nearest_eye_gaze
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from aria_glasses_utils.common import *
from aria_glasses_utils.BetterAriaProvider import BetterAriaProvider, Streams, CustomCalibration


class GazeInference:
    def __init__(self, device="cpu") -> None:
        from aria_glasses_utils.inference.infer import EyeGazeInference

        model = (
            Path(__file__).parent
            / "inference"
            / "model"
            / "pretrained_weights"
            / "social_eyes_uncertainty_v1"
        )
        self._egi = EyeGazeInference(
            model / "weights.pth", model / "config.yaml", device
        )

    def predict(self, img, verbose=False):
        start = time.time()
        preds, lower, upper = self._egi.predict(img)
        if verbose:
            print(f"INFO: Image processed in { time.time()-start} seconds")
        preds = preds.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()
        value_mapping = {
            "yaw": preds[0][0],
            "pitch": preds[0][1],
            "yaw_lower": lower[0][0],
            "pitch_lower": lower[0][1],
            "yaw_upper": upper[0][0],
            "pitch_upper": upper[0][1],
        }
        if verbose == 2:
            print(value_mapping)
        return (value_mapping["yaw"], value_mapping["pitch"])

    def a2t(self, image):
        return torch.tensor(image, device="cpu")

from typing import Dict

class BetterEyeGaze:
    def __init__(
        self,
        custom_calibrations: Dict[Streams, CustomCalibration],
        device_calibrarion: calibration.DeviceCalibration,
        correct_distorsion: bool = True,
        rotate_image: bool = True,
        init_inference=True,
    ) -> None:
        
        
        self.custom_calibrations = custom_calibrations
        self.device_calibrarion = device_calibrarion
        self.correct_distortion = correct_distorsion
        self.rotate_image = rotate_image
        

        self.gaze_inference = None
        if init_inference:
            self.gaze_inference = GazeInference()

    def get_gaze_center(self, gaze_cpf, legacy_model=True):
        if legacy_model:
            return self.get_gaze_center_raw(
                gaze_cpf.yaw, gaze_cpf.pitch, gaze_cpf.depth or 1.0
            )
        else:
            depth, combined_yaw, combined_pitch = (
                mps.compute_depth_and_combined_gaze_direction(
                    gaze_cpf.vergence.left_yaw,
                    gaze_cpf.vergence.right_yaw,
                    gaze_cpf.pitch,
                )
            )
            return self.get_gaze_center_raw(combined_yaw, combined_pitch, depth)

    def get_gaze_center_raw(self, yaw, pitch, depth=1.0):
        """Get the gaze center both in cpf and at depth

        Args:
            yaw: mps output
            pitch: mps output
            depth (float, optional). Defaults to 1.0.

        Returns:
            tuple: gaze_center_in_cpf, gaze_center_in_pixels
        """
        gaze_center_in_cpf = mps.get_eyegaze_point_at_depth(yaw, pitch, depth)
        transform_cpf_sensor = (
            self.device_calibrarion.get_transform_cpf_sensor(
                Streams.RGB.label()
            )
        )
        gaze_center_in_camera = transform_cpf_sensor.inverse() @ gaze_center_in_cpf

        if self.correct_distortion:
            if self.rotate_image:
                gaze_center_in_pixels = (
                    self.custom_calibrations[Streams.RGB]
                    .rotated_pinhole_calib.project(gaze_center_in_camera)
                    .astype(int)
                )
            else:
                gaze_center_in_pixels = (
                    self.custom_calibrations[Streams.RGB]
                    .pinhole_calib.project(gaze_center_in_camera)
                    .astype(int)
                )
        else:
            gaze_center_in_pixels = (
                self.custom_calibrations[Streams.RGB]
                .original_calib.project(gaze_center_in_camera)
                .astype(int)
            )
        
        if self.rotate_image:
            gaze_center_in_pixels = self.rotate_pixel_cw90(gaze_center_in_pixels)


        return gaze_center_in_cpf, gaze_center_in_pixels
    
    def rotate_pixel_cw90(self, gaze_center_in_pixels, scale=1.0):
        return [
            int(self.custom_calibrations[Streams.RGB].img_w * scale) - gaze_center_in_pixels[1],
            gaze_center_in_pixels[0],
        ]

    def getCameraCalib(self):
        if self.correct_distortion:
            if self.rotate_image:
                return self.custom_calibrations[Streams.RGB].rotated_pinhole_calib
            return self.custom_calibrations[Streams.RGB].pinhole_calib
        return self.custom_calibrations[Streams.RGB].original_calib

    def predict(self, img):
        if self.gaze_inference == None:
            raise "Gaze Inference must be initialized before making a prediction"

        if torch.is_tensor(img):
            return self.gaze_inference.predict(img)
        else:
            return self.gaze_inference.predict(self.gaze_inference.a2t(img))


def main():
    config = tomllib.load(open("config.toml", "rb"))
    vrs_file = config["aria_recordings"]["vrs"]
   
    provider = BetterAriaProvider(vrs = vrs_file)

    GENERAL_GAZE = False

    if GENERAL_GAZE:
        eye_gaze_path = config["aria_recordings"]["general_eye_gaze"]
        gaze_cpfs = mps.read_eyegaze(eye_gaze_path)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    
    customCalibrations, calibration_device = provider.get_calibration()

    rotation = True
    undistortion = True


    eye_gaze = BetterEyeGaze(
        custom_calibrations=customCalibrations,
        device_calibrarion=calibration_device,
        correct_distorsion=undistortion,
        rotate_image=rotation,
        init_inference=True,
    )

    for time in provider.get_time_range(100000000):
        img, _ = provider.get_frame(Streams.RGB, time_ns=time, rotated=rotation, undistorted=undistortion)
        img_et, _ = provider.get_frame(Streams.ET, time_ns=time)

        if GENERAL_GAZE:
            gaze_cpf = get_nearest_eye_gaze(gaze_cpfs, time)
            if gaze_cpf is None:
                continue
            gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center(
                gaze_cpf
            )
            cv2.circle(img, gaze_center_in_pixels, 5, (255, 0, 0), 2)

        yaw, pitch = eye_gaze.predict(img_et)
        gaze_center_in_cpf2, gaze_center_in_pixels2 = eye_gaze.get_gaze_center_raw(
            yaw, pitch, 0.5
        )

        cv2.circle(img, gaze_center_in_pixels2, 5, (255, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)

        if quit_keypress():
            break


if __name__ == "__main__":
    main()

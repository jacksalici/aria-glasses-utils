
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from utils import *


from eyeGaze import EyeGaze
from gazeInference import EyeGazeInference
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
    config = tomllib.load(open('config.toml', 'rb'))

    vrs_file = config['aria_recordings'][2]['vrs']
    output_folder = config['aria_recordings'][2]['output']
    
    os.mkdir(output_folder)
    stream_label_rgb = 'camera-rgb'
    provider = data_provider.create_vrs_data_provider(vrs_file)

    stream_id_rgb = provider.get_stream_id_from_label(stream_label_rgb)
    t_first = provider.get_first_time_ns(stream_id_rgb, TimeDomain.DEVICE_TIME)
    t_last = provider.get_last_time_ns(stream_id_rgb, TimeDomain.DEVICE_TIME)

    calib_device = provider.get_device_calibration()

    eye_gaze = EyeGaze(live = False, correct_distorsion=True, rotate_image=True, vrs_file=vrs_file)
    imgs = []
    for time in range(t_first, t_last, 100_000_000):
        print(time)

        img = eye_gaze.get_rgb_image(time_ns=time)
        if len(imgs) == 0:
            imgs.append(img)
        
        if len(imgs)>0 and confidence(img, imgs[-1])<0.7:
            imgs.append(img)
        else:
            if blurryness(img) < blurryness(imgs[-1]):
                imgs[-1] = img 
        
        #cv2.circle(img, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 2)
        #sleep(0.3)
     
    for index, img in enumerate(imgs):
        cv2.imwrite(os.path.join(output_folder,f"img{index}.jpg"), img)   
       
        
if __name__ == "__main__":
    main()
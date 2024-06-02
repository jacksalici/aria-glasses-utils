from projectaria_eyetracking.projectaria_eyetracking.inference.infer import EyeGazeInference
import tomllib
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId

import torch

egi = EyeGazeInference('projectaria_eyetracking/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth',
                       'projectaria_eyetracking/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml')


config = tomllib.load(open('config.toml', 'rb'))

vrsfile = config['aria_recordings'][0]['vrs']
provider = data_provider.create_vrs_data_provider(vrsfile)

from PIL import Image
stream_mappings = {
    #"camera-slam-left": StreamId("1201-1"),
    #"camera-slam-right": StreamId("1201-2"),
    #"camera-rgb": StreamId("214-1"),
    "camera-eyetracking": StreamId("211-1"),
}

index = 1 # sample index (as an example)

image = provider.get_image_data_by_index(stream_mappings['camera-eyetracking'], index)


img = torch.tensor(image[0].to_numpy_array(), device='cpu')

import time
start = time.time()
preds, lower, upper = egi.predict(img)
print("Time", time.time()-start)
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

print(value_mapping)

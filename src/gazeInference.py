import sys
import tomllib

config = tomllib.load(open('config.toml', 'rb'))

sys.path.insert(1, config['general']['projectaria_et_path'])

from inference.infer import EyeGazeInference
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId
from PIL import Image
import torch
import time
import os


class GazeInference():
    def __init__(self) -> None:
        print(config['general']['projectaria_et_path'])
        self._egi = EyeGazeInference(os.path.join(config['general']['projectaria_et_path'], 'inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth'),
                       os.path.join(config['general']['projectaria_et_path'], 'inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml'))

    def predict(self, img, verbose = False):
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
            print(verbose)
        return (value_mapping['yaw'], value_mapping['pitch'])
        
        
def main():
    gaze_inference = GazeInference()
    
    
    vrsfile = config['aria_recordings'][0]['vrs']
    provider = data_provider.create_vrs_data_provider(vrsfile)


    index = 1

    image = provider.get_image_data_by_index(StreamId("211-1"), index)

    img = torch.tensor(image[0].to_numpy_array(), device='cpu')
    
    print(gaze_inference.predict(img))


if __name__ == "__main__":
    main()
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tomllib
    
def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img



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
for [stream_name, stream_id] in stream_mappings.items():
    image = provider.get_image_data_by_index(stream_id, index)
    Image.fromarray(image[0].to_numpy_array()).save(f'{stream_name}.png')
"""    
stream_id = provider.get_stream_id_from_label("imu-left")
accel_x = []
accel_y = []
accel_z = []
gyro_x = []
gyro_y = []
gyro_z = []
timestamps = []
for index in range(0, provider.get_num_data(stream_id)):
    imu_data = provider.get_imu_data_by_index(stream_id, index)
    accel_x.append(imu_data.accel_msec2[0])
    accel_y.append(imu_data.accel_msec2[1])
    accel_z.append(imu_data.accel_msec2[2])
    gyro_x.append(imu_data.gyro_radsec[0])
    gyro_y.append(imu_data.gyro_radsec[1])
    gyro_z.append(imu_data.gyro_radsec[2])
    timestamps.append(imu_data.capture_timestamp_ns * 1e-9)
  
print(len(timestamps))
plt.figure()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"{stream_id.get_name()}")

axes[0].plot(timestamps, accel_x, 'r-', label="x")
axes[0].plot(timestamps, accel_y, 'g-', label="y")
axes[0].plot(timestamps, accel_z, 'b-', label="z")
axes[0].legend(loc='upper left')
axes[0].grid('on')
axes[0].set_xlabel('timestamps (s)')
axes[0].set_ylabel('accelerometer readout (m/sec2)')

axes[1].plot(timestamps, gyro_x, 'r-', label="x")
axes[1].plot(timestamps, gyro_y, 'g-', label="y")
axes[1].plot(timestamps, gyro_z, 'b-', label="z")
axes[1].legend(loc='upper left')
axes[1].grid('on')
axes[1].set_xlabel('timestamps (s)')
axes[1].set_ylabel('gyroscope readout (rad/sec)')

plt.show()
"""
from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tomllib
    

from aria_glasses_utils import utils



config = tomllib.load(open('config.toml', 'rb'))

vrsfile = config['aria_recordings']['vrs']
provider = data_provider.create_vrs_data_provider(vrsfile)

from PIL import Image

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

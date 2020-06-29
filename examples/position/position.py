import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import re
import numpy as np

from pathlib import Path

DEFAULT_DATASET = Path("../../data/outdoor_dataset_02")
IMU_DATA_FILE = DEFAULT_DATASET / "imu_data.txt"

IMU_RE_MASK = r'([0-9]*): accel_x: ([0-9.-]*), ' + \
    'accel_y: ([0-9.-]*), ' + \
    'accel_z: ([0-9.-]*), ' + \
    'gyro_x: ([0-9.-]*), ' + \
    'gyro_y: ([0-9.-]*), ' + \
    'gyro_z: ([0-9.-]*)'

ALPHA_CF = 0.9
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0
FIGSIZE = (15,12)
DPI = 100

class KF():
    def __init__(self):
        self.timestamp_last = None
        self.timestamp = None
        self.dt = None

        # Data dict
        self.data = dict()

        # Observation vector z: [acceleration x with bias, pitch angle]
        self.acc_x = 0.0
        self.pitch = 0.0

        # State vector

        # Other states not part of estimator
        self.roll = 0.0
        self.yaw = 0.0

        # For plotting
        self.timestamp_list = list()
        self.roll_list = list()
        self.pitch_list = list()
        self.yaw_list = list()

    def update(self, data):
        # Update dt
        if self.timestamp is None:
            self.timestamp = data["timestamp"]
            return
        self.timestamp_last = self.timestamp
        self.timestamp = data["timestamp"]
        self.dt = self.timestamp - self.timestamp_last

        # Update complementary filter first with new data
        self.complementary_filter(data)

        # Plotting
        self.timestamp_list.append(self.timestamp)
        self.roll_list.append(self.roll*RAD_TO_DEG)
        self.pitch_list.append(self.pitch*RAD_TO_DEG)
        self.yaw_list.append(self.yaw*RAD_TO_DEG)

    def complementary_filter(self, data):
        # Extract data
        acc_x = data["accel_x"]
        acc_y = data["accel_y"]
        acc_z = data["accel_z"]
        gyro_x = data["gyro_x"]
        gyro_y = data["gyro_y"]
        gyro_z = data["gyro_z"]

        # Estimates
        pitch_accel = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
        roll_accel = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2))

        roll_gyro = gyro_x + \
            gyro_y*np.sin(self.pitch)*np.tan(self.roll) + \
            gyro_z*np.cos(self.pitch)*np.tan(self.roll)

        pitch_gyro = gyro_y*np.cos(self.pitch) - gyro_z*np.sin(self.pitch)

        # Yaw only from gyro
        yaw_gyro = gyro_y*np.sin(self.pitch)*1.0/np.cos(self.roll) + gyro_z*np.cos(self.pitch)*1.0/np.cos(self.roll)

        # Apply complementary filter
        self.pitch = (1.0 - ALPHA_CF)*(self.pitch + pitch_gyro*self.dt) + ALPHA_CF * pitch_accel
        self.roll = (1.0 - ALPHA_CF)*(self.roll + roll_gyro*self.dt) + ALPHA_CF * roll_accel
        self.yaw += yaw_gyro*self.dt

if __name__ == '__main__':
    # Read data from file
    with IMU_DATA_FILE.open('r') as f:
        lines = f.readlines()

    # Empty data list 
    data_list = list()

    # Iterate over all lines
    for line in lines:
        out = re.search(IMU_RE_MASK, line)

        # Assemble dict with current data
        data_now = dict()
        data_now["timestamp"] = int(out.group(1))/1000.0
        data_now["accel_z"] = -float(out.group(2))
        data_now["accel_y"] = float(out.group(3))
        data_now["accel_x"] = float(out.group(4))
        data_now["gyro_z"] = -float(out.group(5))*DEG_TO_RAD
        data_now["gyro_y"] = float(out.group(6))*DEG_TO_RAD
        data_now["gyro_x"] = float(out.group(7))*DEG_TO_RAD

        # Append to data list
        data_list.append(data_now)

    # Create Kalman Filter object
    kf = KF()

    for i in range(len(data_list) - 1):
        # Recursively update estimator
        kf.update(data_list[i])

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    main_p = fig.add_subplot(2, 1, 1)
    main_p.plot(kf.timestamp_list, kf.roll_list, label="roll")
    main_p.plot(kf.timestamp_list, kf.pitch_list, label="pitch")
    main_p.plot(kf.timestamp_list, kf.yaw_list, label="yaw")

    plt.legend()
    plt.show()

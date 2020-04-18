from filterpy.kalman import EKF
from pathlib import Path
from time import sleep
from math import *
from scipy.spatial.transform import Rotation as R

from ..module import Module
from ...utils import DEFAULT_DATASET

LP_LEN = 100
ALPHA_CF = 0.1
DEG_TO_RAD = pi/180.0
RAD_TO_DEG = 180.0/pi
ACCEL_G = -9.80600

class EkfModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(EkfModule, self).__init__(name="ekf_module",
                                            outputs=[("position_vis", 10)],
                                            inputs=["drivers_module:accelerations"], log_dir=log_dir)
        self.args = args

    def start(self):
        # Contains window of raw accel data
        self.imu_data_raw = {var: list() for var in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]}

        # Low pass filtered sensor values
        self.accel_x_lp = 0.0 # IMU Frame low pass filtered accel x data
        self.accel_y_lp = 0.0 # IMU Frame low pass filtered accel y data
        self.accel_z_lp = 0.0 # IMU Frame low pass filtered accel z data
        self.gyro_x_lp = 0.0 # IMU Frame low pass filtered gyro x data
        self.gyro_y_lp = 0.0 # IMU Frame low pass filtered gyro y data
        self.gyro_z_lp = 0.0 # IMU Frame low pass filtered gyro z data

        # Pitch, roll, yaw estimates
        self.pitch_est = 0.0
        self.roll_est = 0.0
        self.yaw_est = 0.0

        # Timestamps
        self.ts_ms = None # Current timestamp
        self.ts_last_ms = 0 # Last timestamp
        self.dt_s = 0 # Current dt

        # World coordinates
        self.world_x = 0
        self.world_y = 0
        self.world_z = 0

        while True:
            sleep(0.05)

            # Get the data
            data = self.get("drivers_module:accelerations")
            if data:
                # Dt calculation
                self.ts_last_ms = self.ts_ms
                self.ts_ms = data["data"]["timestamp"]

                # Handle first timestamp
                if not self.ts_last_ms:
                    self.ts_last_ms = self.ts_ms

                self.dt_s = (self.ts_ms - self.ts_last_ms)/1000.0

                # Low pass filter the data, updates _lp values
                self.apply_low_pass_filters(data["data"])

                # Complementary filter
                self.apply_complementary_filter()

                # Rotate accelerations
                self.estimate_position()

    def estimate_position(self):
        r = R.from_euler('xyz', [-self.roll_est, -self.pitch_est, -self.yaw_est], degrees=False)

        # Rotated accelerations
        accel_w = r.apply([self.accel_z_lp, self.accel_y_lp, self.accel_x_lp])

        self.world_x = self.world_x + accel_w[0]*self.dt_s*self.dt_s
        self.world_y = self.world_y + accel_w[1]*self.dt_s*self.dt_s
        self.world_z = self.world_z + (accel_w[2] - ACCEL_G)*self.dt_s*self.dt_s

        # Publish data
        data_dict = {"pos_x": self.world_x,
                     "pos_y": self.world_y,
                     "pos_z": self.world_z,
                     "angle_x": self.roll_est,
                     "angle_y": self.pitch_est,
                     "angle_z": self.yaw_est}

        self.publish("position_vis", data_dict, 1000, timestamp=self.ts_ms)

    def apply_complementary_filter(self):
        # Pitch and roll based on accel
        pitch_accel = atan(self.accel_y_lp / sqrt(self.accel_z_lp**2 + self.accel_x_lp**2))
        roll_accel = atan(self.accel_z_lp / sqrt(self.accel_y_lp**2 + self.accel_x_lp**2))

        # Pitch, roll and yaw based on gyro
        pitch_gyro = self.gyro_z_lp + \
                     self.gyro_y_lp*sin(self.pitch_est)*tan(self.roll_est) + \
                     self.gyro_x_lp*cos(self.pitch_est)*tan(self.roll_est)

        roll_gyro = self.gyro_y_lp*cos(self.pitch_est) - self.gyro_x_lp*sin(self.pitch_est)

        yaw_gyro = self.gyro_y_lp*sin(self.pitch_est)*1.0/cos(self.roll_est) + self.gyro_x_lp*cos(self.pitch_est)*1.0/cos(self.roll_est)

        # Apply complementary filter
        self.pitch_est = (1.0 - ALPHA_CF)*(self.pitch_est + pitch_gyro*self.dt_s) + ALPHA_CF * pitch_accel
        self.roll_est = (1.0 - ALPHA_CF)*(self.roll_est + roll_gyro*self.dt_s) + ALPHA_CF * roll_accel
        self.yaw_est = yaw_gyro

    def apply_low_pass_filters(self, data):
        # Low pass filter accel data
        self.accel_x_lp = self.low_pass(data["accel_x"], self.imu_data_raw["accel_x"])
        self.accel_y_lp = self.low_pass(data["accel_y"], self.imu_data_raw["accel_y"])
        self.accel_z_lp = self.low_pass(data["accel_z"], self.imu_data_raw["accel_z"])

        # Low pass filter
        self.gyro_x_lp = self.low_pass(data["gyro_x"], self.imu_data_raw["gyro_x"], is_angle=True)
        self.gyro_y_lp = self.low_pass(data["gyro_y"], self.imu_data_raw["gyro_y"], is_angle=True)
        self.gyro_z_lp = self.low_pass(data["gyro_z"], self.imu_data_raw["gyro_z"], is_angle=True)

    def low_pass(self, val, data_raw, max_len=LP_LEN, is_angle=False):
        val = float(val)

        if is_angle:
            val = val*DEG_TO_RAD

        # Only keep LP_LEN data points
        if len(data_raw) > max_len:
            del data_raw[0]
        data_raw.append(val)

        # Calculate mean of current window
        val_lp = 0.0
        for v in data_raw:
            val_lp += v
        val_lp /= len(data_raw)

        return val_lp

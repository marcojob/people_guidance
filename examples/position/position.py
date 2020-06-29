import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import re
import numpy as np

from pathlib import Path

DEFAULT_DATASET = Path("../../data/outdoor_dataset_04")
IMU_DATA_FILE = DEFAULT_DATASET / "imu_data.txt"

IMU_RE_MASK = r'([0-9]*): accel_x: ([0-9.-]*), ' + \
    'accel_y: ([0-9.-]*), ' + \
    'accel_z: ([0-9.-]*), ' + \
    'gyro_x: ([0-9.-]*), ' + \
    'gyro_y: ([0-9.-]*), ' + \
    'gyro_z: ([0-9.-]*)'

ALPHA_CF = 0.5
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0
FIGSIZE = (15,12)
DPI = 100
G = -9.80600
N_STATES = 4
N_MEAS = 2

class KF():
    def __init__(self):
        self.timestamp_last = None
        self.timestamp = None
        self.dt = None

        # Data dict
        self.data = dict()

        # Observation vector z: [acceleration x with bias, pitch angle]
        self.z_meas = np.zeros((N_MEAS, 1))

        # State vector x: [pos x, vel x, acc x, pitch]
        self.x_prio = np.zeros((N_STATES, 1))
        self.x_post = np.zeros((N_STATES, 1))

        # Covariance matrices
        self.P_prio = np.zeros((N_STATES, N_STATES))
        self.P_post = np.zeros((N_STATES, N_STATES))

        # Kalman gain
        self.K = np.zeros((N_STATES, N_MEAS))

        # Process variance matrix
        self.Q = np.diag([0.05, 0.005, 0.0005, 0.08])

        # Measurement noise
        self.R = np.diag([0.0005, 0.08])

        # Model
        self.A = np.array([[1.0, self.dt, 0.0, 0.0],
                           [0.0, 1.0, self.dt, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

        self.H = np.array([[0.0, 0.0, 1.0, G],
                           [0.0, 0.0, 0.0, 1.0]])

        # Other states not part of estimator
        self.roll = 0.0
        self.yaw = 0.0

        # For plotting
        self.timestamp_list = list()
        self.roll_list = list()
        self.pitch_list = list()
        self.yaw_list = list()

        self.pos_x_list = list()
        self.pos_vx_list = list()
        self.pos_ax_list = list()
        self.pos_pitch_list = list()

        self.raw_list = list()
        self.raw2_list = list()
        self.post_list = list()
        self.prio_list = list()

    def update(self, data):
        # Update dt
        if self.timestamp is None:
            self.timestamp = data["timestamp"]
            return
        self.timestamp_last = self.timestamp
        self.timestamp = data["timestamp"]
        self.dt = self.timestamp - self.timestamp_last

        # Update complementary filter first with new data
        self.z_meas[0] = data["accel_x"]
        self.complementary_filter(data)

        # Update A, time-varying
        self.A = self.get_A(self.dt)

        # Prior update
        self.x_prio = np.dot(self.A, self.x_post)
        self.P_prio = np.dot(np.dot(self.A, self.P_post), self.A.T) + self.Q

        # Posterior update
        inv = np.linalg.pinv(np.dot(np.dot(self.H, self.P_prio), self.H.T) + self.R)
        self.K = np.dot(np.dot(self.P_prio, self.H.T), inv) # Kalman gain

        innovation = self.z_meas - np.dot(self.H, self.x_prio) # Innovation matrix

        self.x_post = self.x_prio + np.dot(self.K, innovation)

        """
        if self.x_post[1] > 1:
            self.x_post[1] = 1
        elif self.x_post[1] < -1:
            self.x_post[1] = -1
        """

        temp = np.eye(N_STATES) - np.dot(self.K, self.H)
        self.P_post = np.dot(np.dot(temp, self.P_prio), temp.T) + np.dot(np.dot(self.K, self.R), self.K.T)

        # Plotting
        self.timestamp_list.append(self.timestamp)
        self.roll_list.append(self.roll*RAD_TO_DEG)
        self.pitch_list.append(self.z_meas[1]*RAD_TO_DEG)
        self.yaw_list.append(self.yaw*RAD_TO_DEG)

        self.pos_x_list.append(self.x_post[0])
        self.pos_vx_list.append(self.x_post[1])
        self.pos_ax_list.append(self.x_post[2])
        self.pos_pitch_list.append(self.x_post[3])

        #self.raw_list.append(data["accel_x"] - np.sin(self.z_meas[1])*G)
        self.raw_list.append(np.sin(self.z_meas[1])*G)
        self.raw2_list.append(data["accel_x"])
        self.post_list.append(self.x_post[2])
        self.prio_list.append(self.x_prio[2])

    def get_A(self, dt):
        # Actually have a time varying A matrix
        A = np.array([[1.0, dt, 0.0, 0.0],
                      [0.0, 1.0, dt, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        return A

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
            gyro_y*np.sin(self.z_meas[1])*np.tan(self.roll) + \
            gyro_z*np.cos(self.z_meas[1])*np.tan(self.roll)

        pitch_gyro = gyro_y*np.cos(self.z_meas[1]) - gyro_z*np.sin(self.z_meas[1])

        # Yaw only from gyro
        yaw_gyro = gyro_y*np.sin(self.z_meas[1])*1.0/np.cos(self.roll) + gyro_z*np.cos(self.z_meas[1])*1.0/np.cos(self.roll)

        # Apply complementary filter
        self.z_meas[1] = (1.0 - ALPHA_CF)*(self.z_meas[1] + pitch_gyro*self.dt) + ALPHA_CF * pitch_accel
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
    main_p.plot(kf.timestamp_list, kf.raw_list, label="expected g")
    main_p.plot(kf.timestamp_list, kf.raw2_list, label="measured a")
    main_p.plot(kf.timestamp_list, kf.post_list, label="post")
    main_p.plot(kf.timestamp_list, kf.prio_list, label="prio")

    main_p.legend()

    main_s = fig.add_subplot(2, 1, 2)
    main_s.plot(kf.timestamp_list, kf.pos_x_list, label="pos x")
    main_s.plot(kf.timestamp_list, kf.pos_vx_list, label="vel x")
    main_s.plot(kf.timestamp_list, kf.pos_ax_list, label="acc x")
    main_s.plot(kf.timestamp_list, kf.pos_pitch_list, label="pitch")

    main_s.set_ylim((-10,10))

    main_s.legend()
    plt.show()

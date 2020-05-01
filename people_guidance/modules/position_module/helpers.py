import collections
from math import pi, atan, sqrt, sin, cos, tan
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

from .cam import cam

from math import atan2, sqrt, cos, sin

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])
VOResult = collections.namedtuple("VOResult", ["homogs", "pairs", "ts0", "ts1", "image"])


DEGREE_TO_RAD = float(pi / 180)


def degree_to_rad(angle: float) -> float:
    return angle * DEGREE_TO_RAD


def interpolate_frames(frame0, frame1, ts: int):
    assert frame0.ts <= ts <= frame1.ts, "Timestamp for interpolation must be in between frame0.ts and frame1.ts"

    lever = (ts - frame0.ts) / (frame1.ts - frame0.ts)

    properties = {"ts": ts}
    for key in ["ax", "ay", "az", "gx", "gy", "gz"]:
        value = getattr(frame0, key) + ((getattr(frame1, key) - getattr(frame0, key)) * lever)
        properties[key] = value
    return IMUFrame(**properties)


class MovingAverageFilter:
    def __init__(self):
        self.keys: Dict[str, List] = {}

    def __call__(self, key: str, value: Union[int, float], window_size: int = 5):
        if key not in self.keys:
            self.keys[key] = [value]
        else:
            self.keys[key].append(value)
            while len(self.keys[key]) > window_size:
                self.keys[key].pop(0)

        return float(sum(self.keys[key]) / len(self.keys[key]))

class Homography:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                 rotation_matrix=np.identity(3)):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.roll: float = roll
        self.pitch: float = pitch
        self.yaw: float = yaw
        self.rotation_matrix: np.array[float] = rotation_matrix

    def __str__(self):
        return f"Homography: (translation ({self.x}, {self.y}, {self.z}), angles ({self.roll}, {self.pitch}, {self.yaw}))"

    def as_matrix(self) -> np.array:
        translation = np.array((self.x, self.y, self.z))
        return np.column_stack((self.rotation_matrix, translation))


class Pose:
    def __init__(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class ComplementaryFilter:
    def __init__(self, alpha: float = 0.2):
        self.last_frame = None
        self.pose = Pose()
        self.alpha = alpha

    def __call__(self, frame: IMUFrame) -> IMUFrame:
        if self.last_frame is None:
            self.last_frame = frame
            # return directly with simple gravity subtraction
        else:
            dt = self.last_frame.ts - frame.ts
            # complementary filter

            # 1. Acceleration component, in RADIAN, cannot work because of the singularity issue
            # roll_accel = atan2(-frame.ay, sqrt(frame.ax ** 2 + frame.az ** 2))
            # pitch_accel = atan2(frame.ax, sqrt(frame.ay ** 2 + frame.az ** 2))

            # 1. Using quaternion to represent the rotation from the -x Axis to the 'gravity' vector
            v1 = np.array([0, 0, -1]) # negative z axis corresponds to the gravity vector when in the initial state
            v2_not_normalised = np.array([frame.ax, frame.ay, frame.az]) # gravity vector
            v2_norm = norm(v2_not_normalised)
            v2 = v2_not_normalised / v2_norm # will very unlikely be a division by 0
            q = np.concatenate((np.cross(v1, v2), np.array([sqrt((norm(v1) ** 2) * (norm(v2) ** 2)) + np.dot(v1, v2)])))
            # Rotation from -Z axis to gravity vector
            r_to_acc_vector = Rotation.from_quat(q)

            q_second = self.q_from_acc(frame.ax, frame.ay, frame.az)
            r_to_acc_vector2 = Rotation.from_quat(q_second)

            cam(q)
            print(f"Acceleration: {[frame.ax, frame.ay, frame.az]}"
                  f"first q: \n{q}, second: \n{q_second}, difference of rot: \n"
                  f"first : \n{r_to_acc_vector.as_matrix()}, first inv \n{r_to_acc_vector.inv().as_matrix()}, "
                  f"mult \n{np.dot(r_to_acc_vector.as_matrix(), r_to_acc_vector.inv().as_matrix())}"
                  f"second : \n{r_to_acc_vector2.as_matrix()}")

            # 2. Integrate the

            # https: // github.com / jrowberg / i2cdevlib / blob / master / Arduino / MPU6050 / MPU6050_6Axis_MotionApps20.h
            # uint8_t
            # MPU6050::dmpGetYawPitchRoll(float * data, Quaternion * q, VectorFloat * gravity)
            # {
            # // yaw: (about Z axis)
            # data[0] = atan2(2 * q -> x * q -> y - 2 * q -> w * q -> z, 2 * q -> w * q -> w + 2 * q -> x * q -> x - 1);
            # // pitch: (nose up / down, about Y axis)
            # data[1] = atan2(gravity -> x, sqrt(gravity -> y * gravity -> y + gravity -> z * gravity -> z));
            # // roll: (tilt left / right, about X axis)
            # data[2] = atan2(gravity -> y, gravity -> z);
            # if (gravity -> z < 0) {
            # if (data[1] > 0) {
            # data[1] = PI - data[1];
            # } else {
            # data[1] = -PI - data[1];
            # }
            # }
            # return 0;
            # }
            #
            # # Gravity vector [vx, vy, vz] from quaternion q [qw, qx, qy, qz]
            # vx = 2 * (qx * qz - qw * qy)
            # vy = 2 * (qw * qx + qy * qz)
            # vz = qw * qw - qx * qx - qy * qy + qz * qz
            #
            # # Euler (data) from quaternion q
            # data[0] = atan2(
            #     2 * q -> x * q -> y - 2 * q -> w * q -> z, 2 * q -> w * q -> w + 2 * q -> x * q -> x - 1) # psi
            # data[1] = -asin(2 * q -> x * q -> z + 2 * q -> w * q -> y) # theta
            # data[2] = atan2(
            #     2 * q -> y * q -> z - 2 * q -> w * q -> x, 2 * q -> w * q -> w + 2 * q -> z * q -> z - 1) # phi


            # # Pitch, roll and yaw based on gyro
            # roll_gyro = frame.gz + \
            #             frame.gy * sin(self.pose.pitch) * tan(self.pose.roll) + \
            #             frame.gx * cos(self.pose.pitch) * tan(self.pose.roll)
            #
            # pitch_gyro = frame.gy * cos(self.pose.pitch) - frame.gx * sin(self.pose.pitch)
            #
            # yaw_gyro = frame.gy * sin(self.pose.pitch) * 1.0 / cos(self.pose.roll) + frame.gx * cos(
            #     self.pose.pitch) * 1.0 / cos(self.pose.roll)
            #
            # # Apply complementary filter
            # self.pose.roll = (1.0 - self.alpha) * (self.pose.roll + roll_gyro * dt) + self.alpha * roll_accel
            # self.pose.pitch = (1.0 - self.alpha) * (self.pose.pitch + pitch_gyro * dt) + self.alpha * pitch_accel
            # self.pose.yaw += yaw_gyro * dt #* 0.9

        # TODO: Use this new pose estimate to remove gravity acc from frame, return the frame.
        return IMUFrame( # Need to compute that
            ax=frame.ax,
            ay=frame.ay,
            az=frame.az + 9.8,
            gx=frame.gx,
            gy=frame.gy,
            gz=frame.gz,
            ts=frame.ts
        )

    #https://github.com/ccny-ros-pkg/imu_tools/blob/indigo/imu_complementary_filter/src/complementary_filter.cpp
    def q_from_acc(self, ax : float, ay: float, az: float):
        # input: acceleration vector
        # output: q0_meas, q1_meas, q2_meas, q3_meas
        # q_acc is the quaternion obtained from the acceleration vector representing the orientation of the Global frame
            # wrt the Local frame with arbitrary yaw (intermediary frame).q3_acc is defined as 0.

        # Normalize acceleration vector
        [ax, ay, az] = np.array([ax, ay, az]) / norm([ax, ay, az])

        if (az >= 0):
            q0_meas = sqrt((az + 1) * 0.5)
            q1_meas = -ay / (2.0 * q0_meas)
            q2_meas = ax / (2.0 * q0_meas)
            q3_meas = 0
        else:
            x = sqrt((1 - az) * 0.5)
            q0_meas = -ay / (2.0 * x)
            q1_meas = x
            q2_meas = 0
            q3_meas = ax / (2.0 * x)

        return [q0_meas, q1_meas, q2_meas, q3_meas]


def visualize_input_data(frames: List[IMUFrame]) -> None:
    #PLOT
    plt.figure(3, figsize=(10, 12))
    plt.tight_layout()

    plt.suptitle(f"Input data", x=0.5, y=.999)

    plt.subplot(3, 1, 1)
    plt.scatter([i for i in range(len(frames))], [frames[i].gx for i in range(len(frames))])
    plt.title('gx')
    plt.xlabel('')
    plt.ylabel('')

    plt.subplot(3, 1, 2)
    plt.scatter([i for i in range(len(frames))], [frames[i].gy for i in range(len(frames))])
    plt.title('gy')
    plt.xlabel('')
    plt.ylabel('')

    plt.subplot(3, 1, 3)
    plt.scatter([i for i in range(len(frames))], [frames[i].gz for i in range(len(frames))])
    plt.title('gz')
    plt.xlabel('')
    plt.ylabel('')

    plt.pause(0.001)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(counter=0)
def visualize_distance_metric(best_match, best_match2, degrees, imu_angles, vo_angles) -> None:
    visualize_distance_metric.counter += 1
    plt.figure(1, figsize=(10, 12))
    plt.tight_layout()

    plt.suptitle(f"Distance evaluation method", x=0.5, y=.999)

    plt.subplot(2, 1, 1)
    plt.scatter(visualize_distance_metric.counter, best_match[1])
    plt.title('Distance Theo')
    plt.xlabel('')
    plt.ylabel('')

    plt.subplot(2, 1, 2)
    plt.scatter(visualize_distance_metric.counter, best_match2[1])
    plt.title('Distance Adrian')
    plt.xlabel('')
    plt.ylabel('')

    visualize_distance_metric.counter += 1
    plt.pause(0.001)

    # #PLOT
    # plt.figure(2, figsize=(10, 12))
    # plt.tight_layout()
    #
    # plt.suptitle(f"Evolution of the rotations, degrees? {degrees}", x=0.5, y=.999)
    #
    # plt.subplot(2, 1, 1)
    # plt.scatter(visualize_distance_metric.counter, imu_angles[2])
    # plt.title('rot IMU')
    # plt.xlabel('')
    # plt.ylabel('')
    #
    # plt.subplot(2, 1, 2)
    # plt.scatter(visualize_distance_metric.counter, vo_angles[2])
    # plt.title('rot VO')
    # plt.xlabel('')
    # plt.ylabel('')

    plt.pause(0.001)

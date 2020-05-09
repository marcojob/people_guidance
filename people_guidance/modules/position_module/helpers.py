import collections
from math import pi, atan, sqrt, sin, cos, tan
from time import sleep
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

from .cam import CameraPygame

from math import atan2, sqrt, cos, sin

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])
VOResult = collections.namedtuple("VOResult", ["homogs", "pairs", "ts0", "ts1", "image"])

DEGREE_TO_RAD = float(pi / 180)


def degree_to_rad(angle: float) -> float:
    return angle * DEGREE_TO_RAD

def rad_to_degrees(angle: float) -> float:
    return angle / DEGREE_TO_RAD


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

    def __call__(self, key: str, value: Union[int, float], window_size: int = 5, median=False, n_median=2):
        if key not in self.keys:
            self.keys[key] = [value]
        else:
            self.keys[key].append(value)
            while len(self.keys[key]) > window_size:
                self.keys[key].pop(0)

            #implement a median filter taking out the largest n_median values
            if median and window_size > n_median * 5:
                for i in range(n_median):
                    try:
                        self.keys[key].remove(max(abs(self.keys[key])))
                    except:
                        self.keys[key].remove(-max(abs(self.keys[key])))


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
    # https://www.mdpi.com/1424-8220/15/8/19302/htm
    def __init__(self, alpha: float = 0.2):
        self.last_frame = None
        self.pose = Pose()
        self.alpha = alpha
        self.cam = CameraPygame()
        self.q_gyro_state = [1, 0, 0, 0]  # Local camera frame in inertial frame init
        # Debug
        self.yaw = 0

    def __call__(self, frame: IMUFrame, alpha=0.5) -> IMUFrame:
        if self.last_frame is None:
            self.last_frame = frame
            return IMUFrame(
                ax=frame.ax,
                ay=frame.ay,
                az=frame.az + 9.8,
                gx=frame.gx,
                gy=frame.gy,
                gz=frame.gz,
                ts=frame.ts
            )
        else:
            dt = (frame.ts - self.last_frame.ts)/1000
            # complementary filter

            # 1. Acceleration component, in RADIAN, cannot work because of the singularity issue
            # roll_accel = atan2(-frame.ay, sqrt(frame.ax ** 2 + frame.az ** 2))
            # pitch_accel = atan2(frame.ax, sqrt(frame.ay ** 2 + frame.az ** 2))

            # 1. Using quaternion to represent the rotation to the gravity vector
            # A. from the -Z Axis to the 'gravity' vector
            # # two constraints, 3 DOF
            # q_old = self.q_from_acc0(frame.ax, frame.ay, frame.az)
            # r_to_acc_vector = Rotation.from_quat(q_old)

            # B. Quaternion estimation using paper from 2015
            q_acc = self.q_from_acc2(frame.ax, frame.ay, frame.az) # Rotation q_AI : inertial frame represented in IMU frame
            # r_to_acc_vector2 = Rotation.from_quat(q_acc)
            # To recreate [0, 0, -g], apply this formula:
            # quaternion_apply(quaternion_conjugate(q_acc), [frame.ax, frame.ay, frame.az])[1:]

            # # 2. Gyroscope angular speed to quaternion state update
            # # A. Source: Quaternion kinematics for the error-state Kalman Filter, Joan Sola, November 8, 2017. ~p.49
            gyro = np.array([frame.gx, frame.gy, frame.gz])
            gyro_norm = norm(gyro)
            # gyro /= gyro_norm
            q_gyro = np.array([1, 0, 0, 0])
            if gyro_norm > 0.0000001:
                q_gyro = np.concatenate((np.array([cos(gyro_norm * dt * 0.5)]),
                                         gyro/gyro_norm * sin(gyro_norm * dt * 0.5)), axis=0)
                # q_gyro = np.concatenate((np.array([cos(gyro_norm * dt * 0.5)]),
                #                                 gyro / gyro_norm * sin(gyro_norm * dt * 0.5)))  # (214)
            # q_gyro = np.concatenate((np.array([0]),
            #                                 gyro))  # (214)
            q_gyro /= norm(q_gyro)

            self.q_gyro_state = quaternion_multiply(self.q_gyro_state, q_gyro)  # (211)

            # # B. Source: Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs
            # # Roberto G. Valenti et al ~p15
            # gyro = np.array([frame.gx, frame.gy, frame.gz]) * DEGREE_TO_RAD
            # gyro_norm = norm(gyro)
            # q_gyro = np.array([1, 0, 0, 0])
            # if gyro_norm > 0.0000001:
            #     q_gyro = np.concatenate((np.array([cos(gyro_norm * dt * 0.5)]),
            #                                     gyro * sin(gyro_norm * dt * 0.5)))
            #
            # # # q_gyro = np.concatenate((np.array([0]), gyro))  # (37)
            # q_dot = 0.5 * quaternion_multiply(self.q_state, q_gyro) # (37)
            # self.q_state += q_dot * dt # (42)
            # self.q_state /= norm(self.q_state)
            # # Correction step
            # alpha_lerp = 0.999 # gain that characterizes the cut-off frequency of the filter [35]
            # delta_q_acc_bar = (1-alpha_lerp) * np.array([1, 0, 0, 0]) + alpha_lerp * q_acc # LERP (50)
            # delta_q_acc_hat = delta_q_acc_bar / norm(delta_q_acc_bar) # LERP (51)
            # self.q_state = quaternion_multiply(self.q_state, delta_q_acc_hat) # (43)

            # # C. Source: https://www.thepoorengineer.com/en/quaternion/
            # gyro = np.array([frame.gx, frame.gy, frame.gz]) * DEGREE_TO_RAD
            # q = self.q_state
            # Sq = np.array([[-q[1], -q[2], -q[3]],
            #                [q[0], -q[3], q[2]],
            #                [q[3], q[0], -q[1]],
            #                [-q[2], q[1], q[0]]])
            # q_update = dt / 2 * np.matmul(Sq, np.array(gyro).transpose())
            # self.q_state += q_update
            # self.q_state /= norm(self.q_state)

            # # D.
            # # https://stackoverflow.com/questions/12053895/converting-angular-velocity-to-quaternion-in-opencv
            # gyro = np.array([frame.gx, frame.gy, frame.gz]) * DEGREE_TO_RAD
            # gyro_norm = norm(gyro)
            # # gyro /= gyro_norm
            # q_gyro_update = np.array([1, 0, 0, 0])
            # if gyro_norm > 0.0000001:
            #     q_gyro_update = np.concatenate((np.array([cos(gyro_norm * dt * 0.5)]),
            #                                     gyro * sin(gyro_norm * dt * 0.5)))
            #
            # self.q_state = quaternion_multiply(self.q_state, q_gyro_update)  # (211)

            # 3. Linear interpolation
            # # A. Using slerp interpolation
            # # TODO: for yaw, only consider data from gyro
            # # 3.1 extract the yaw from the gyro data
            # [gyro_yaw, gyro_pitch, gyro_roll] = self.yaw_from_q(self.q_gyro_state)
            # # 3.2 set the yaw in the q_acc equal to gyro_yaw
            # # ??
            # # equal to the first argument for t_=0, to the second for t_=1
            # self.q_state = self.slerp(self.q_gyro_state, q_acc, t_=0)

            # B. Directly interpolating the Y, P, R
            # 3.1 get the angles
            [gyro_yaw, gyro_pitch, gyro_roll] = quat_to_ypr(self.q_gyro_state)
            [accel_yaw, accel_pitch, accel_roll] = quat_to_ypr(q_acc)


            alpha = 1
            # 3.2 complementary filter considering that the gyro yaw is correct
            # RADIAN
            [yaw, pitch, roll] = np.array([gyro_yaw, gyro_pitch, gyro_roll]) * (1-alpha)  + \
                                 np.array([gyro_yaw, accel_pitch, accel_roll]) * alpha
            # 3.3 Save to quaternion and update state
            self.q_gyro_state = ypr_to_quat(ypr=[yaw, pitch, roll])

            # print(f"current gyro data : {gyro}")
            # print(f"current q_state_gyro : {self.q_state}")
            # print(f"q_update: {q_gyro}")

            # Pygames visualization
            self.cam(self.q_gyro_state, name=f"q_update, time = {frame.ts/1000}, yaw = {self.yaw}")
            # self.cam(np.array([yaw, pitch, roll]) / DEGREE_TO_RAD, name=f"q_update, time = {frame.ts/1000}, dt = {dt}", useQuat=False) # in case of not passing quaternion: [yaw, pitch, roll] in DEGREES

            # print(f"Acceleration: {[frame.ax, frame.ay, frame.az]}"
            #       f"first q: \n{q}, second: \n{q_second}, difference of rot: \n"
            #       f"first : \n{r_to_acc_vector.as_matrix()}, first inv \n{r_to_acc_vector.inv().as_matrix()}, "
            #       f"mult \n{np.dot(r_to_acc_vector.as_matrix(), r_to_acc_vector.inv().as_matrix())}"
            #       f"second : \n{r_to_acc_vector2.as_matrix()}")

            # 4. Express the gravity vector in the local frame
            # To recreate [0, 0, -g], apply this formula:
            local_gravity = quaternion_apply(self.q_gyro_state, [0, 0, -1])[1:] * 9.81
            print(f"gravity compensation: {np.array([frame.ax, frame.ay, frame.az]) - local_gravity}")

            #Update before returning
            self.last_frame = frame

            # TODO: Use this new pose estimate to remove gravity acc from frame, return the frame.
            return IMUFrame(  # Need to compute that
                ax=frame.ax - local_gravity[0],
                ay=frame.ay - local_gravity[1],
                az=frame.az - local_gravity[2],
                gx=frame.gx,
                gy=frame.gy,
                gz=frame.gz,
                ts=frame.ts
            )

    def q_from_acc0(self, ax, ay, az):
        v1 = np.array([0, 0, -1])  # negative z axis corresponds to the gravity vector when in the initial state
        v2_not_normalised = np.array([ax, ay, az])  # gravity vector
        v2_norm = norm(v2_not_normalised)
        v2 = v2_not_normalised / v2_norm  # will very unlikely be a division by 0
        return np.concatenate((np.cross(v1, v2), np.array([sqrt((norm(v1) ** 2) * (norm(v2) ** 2)) + np.dot(v1, v2)])))

    # https://github.com/ccny-ros-pkg/imu_tools/blob/indigo/imu_complementary_filter/src/complementary_filter.cpp
    def q_from_acc(self, ax: float, ay: float, az: float) -> List:
        # input: acceleration vector
        # output: q0_meas, q1_meas, q2_meas, q3_meas
        # q_acc is the quaternion obtained from the acceleration vector representing the orientation of the Global frame
        # wrt the Local frame with arbitrary yaw (intermediary frame).q3_acc is defined as 0.

        # Normalize acceleration vector
        [ax, ay, az] = np.array([ax, ay, az]) / norm([ax, ay, az])
        q_vector = [0, 0, 0, 0]

        if (az >= 0):
            q_vector[0] = sqrt((az + 1) * 0.5)
            q_vector[1] = -ay / (2.0 * q_vector[0])
            q_vector[2] = ax / (2.0 * q_vector[0])
            q_vector[3] = 0
        else:
            x = sqrt((1 - az) * 0.5)
            q_vector[0] = -ay / (2.0 * x)
            q_vector[1] = x
            q_vector[2] = 0
            q_vector[3] = ax / (2.0 * x)

        return np.array(q_vector)

    def q_from_acc2(self, ax: float, ay: float, az: float) -> List:
        '''
        https://www.mdpi.com/1424-8220/15/8/19302/htm
        set one element of the quaternion to 0 to et a fully defined system of equations
        not continuous at the az=0 point. Should use a magnetometer to refine the solution

        The calculation occurs as if the x coordinate system was showing towards the earth. This is not the case and
        a correction needs to be implemented.
        Compensate by setting the acceleration vector to its opposite (*-1)

        :param ax: acceleration x
        :param ay: acceleration y
        :param az: acceleration z
        :return: list forming the quaternion
        '''
        # input: acceleration vector
        # output: q0_meas, q1_meas, q2_meas, q3_meas
        # q_acc is the quaternion obtained from the acceleration vector representing the orientation of the Global frame
        # wrt the Local frame with arbitrary yaw (intermediary frame).q3_acc is defined as 0.

        # Normalize acceleration vector
        [ax, ay, az] = - np.array([ax, ay, az]) / norm([ax, ay, az])  # Minus due to inverted gravity direction
        q_vector = [0, 0, 0, 0]

        if (az >= 0):
            x = sqrt(2 * (az + 1))
            q_vector[0] = x / 2
            q_vector[1] = -ay / x
            q_vector[2] = ax / x
            q_vector[3] = 0
        else:  # singularity when approaching az = -1
            x = sqrt(2 * (1 - az))
            q_vector[0] = -ay / x
            q_vector[1] = x / 2
            q_vector[2] = 0
            q_vector[3] = ax / x

        # TODO: subtract the yaw since g does not give any information regarding the yaw

        return np.array(q_vector)

    def slerp(self, v0, v1, t_=0):
        """Spherical linear interpolation.
        https://en.wikipedia.org/wiki/Slerp
        """
        # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
        v0 = np.array(v0)
        v0 /= norm(v0)
        v1 = np.array(v1)
        v1 /= norm(v1)
        dot = np.sum(v0 * v1)

        if dot < 0.0:
            v1 = -v1
            dot = -dot

        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            result = v0 + t_ * (v1 - v0)
            return result / norm(result)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * t_
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * v0) + (s1 * v1)

def quat_to_ypr(q):
    # Output in RAD
    yaw = atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
    pitch = -sin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll = atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
    return np.array([yaw, pitch, roll])

def ypr_to_quat(ypr): # yaw (Z), pitch (Y), roll (X)
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    [yaw, pitch, roll] = ypr
    # Abbreviations for the various angular functions
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.array([w, x, y, z])

    return q


# Quaternion kinematics for the error-state Kalman Filter, Joan Sola, November 8, 2017
def quaternion_multiply(quaternion1, quaternion2):
    w0, x0, y0, z0 = quaternion2 / norm(quaternion2)
    w1, x1, y1, z1 = quaternion1 / norm(quaternion1)
    output = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    return output / norm(output)

def quaternion_apply(quaternion : List, vector : List):
    q2 = np.concatenate((np.array([0.0]), np.array(vector)))
    return quaternion_multiply(quaternion_multiply(quaternion, q2),
                               quaternion_conjugate(quaternion))

def quaternion_conjugate(quaternion):
    w, x, y, z = quaternion
    return np.array([w, -x, -y, -z])


def visualize_input_data(frames: List[IMUFrame]) -> None:
    # PLOT
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

import collections
import sys
from math import pi, atan, sqrt, sin, cos, tan
from time import sleep
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

from math import atan2, sqrt, cos, sin
from cmath import acos

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "quaternion", "ts"])
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

    # interpolate quaternion
    q0 = np.array(frame0.quaternion)
    q1 = np.array(frame1.quaternion)
    properties["quaternion"] = nlerp(q0, q1, lever)

    return IMUFrame(**properties)


class Velocity:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def dampen(self, mu=0.95):
        self.x *= mu
        self.y *= mu
        self.z *= mu


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

            # median filter taking out the largest n_median values
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
        # acceleration
        self.x: float = x
        self.y: float = y
        self.z: float = z
        # angle axis
        self.roll: float = roll
        self.pitch: float = pitch
        self.yaw: float = yaw
        # rotation matrix
        # Check if orthogonal and determinant = 1
        check_correct_rot_mat(rotation_matrix)
        self.rotation_matrix: np.array[float] = rotation_matrix

    def __str__(self):
        return f"Homography: (translation ({self.x}, {self.y}, {self.z}), angles ({self.roll}, {self.pitch}, {self.yaw}))"

    def as_Tmatrix(self) -> np.array:
        translation = np.array((self.x, self.y, self.z))
        return np.column_stack((self.rotation_matrix, translation))


def check_correct_rot_mat(rotation_matrix) -> None:
    if (np.abs(np.round(rotation_matrix.dot(rotation_matrix.T), 2)) == np.eye(3, 3)).all() and abs(
            np.round(np.linalg.det(rotation_matrix), 2)) == 1:
        pass
    else:
        sys.exit(f'rotation matrix is no orthogonal matrix, {rotation_matrix}, det: {np.linalg.det(rotation_matrix)}, mat: {rotation_matrix.dot(rotation_matrix.T)}')


def normalise_rotation(rot, error=1e-6):
    # re-normalize
    # https://math.stackexchange.com/questions/3292034/normalizing-a-rotation-matrix
    try:
        rot = cay(skewRot(cay(rot)))
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        while abs(np.linalg.det(rot) - 1.0) > error:
            rot = 3. / 2. * rot - 0.5 * rot.dot(rot.T.dot(rot))  # iterative method
    return rot


def cay(rot):
    # with rot being any skew symmetric matrix => I + A is invertible
    # https://en.wikipedia.org/wiki/Cayley_transform
    return (np.eye(3) - rot).dot(np.linalg.inv(np.eye(3) + rot))


class Pose:
    def __init__(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

class pygameVisualize:
    def __init__(self, alpha: float = 0.2):
        self.visualize = False
        if self.visualize:
            self.cam = CameraPygame()

    def __call__(self, quat, visualize=False, name="Quaternion displayed"):
        if visualize and self.visualize:
            self.cam(quat, name=name)

class ComplementaryFilter:
    # https://www.mdpi.com/1424-8220/15/8/19302/htm
    def __init__(self, alpha: float = 0.2):
        self.last_frame = None
        self.pose = Pose()
        self.alpha = alpha

        self.visualize = False
        if self.visualize:
            self.cam = CameraPygame()
        self.q_gyro_state = [1, 0, 0, 0]  # Local camera frame in inertial frame init

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
                quaternion=[1, 0, 0, 0],
                ts=frame.ts
            )
        else:
            dt = (frame.ts - self.last_frame.ts) / 1000
            # complementary filter

            # 1. Acceleration component, in RADIAN, cannot work because of the singularity issue
            q_acc = self.q_from_acc2(frame.ax, frame.ay,
                                     frame.az)  # Rotation q_AI : inertial frame represented in IMU frame

            # 2. Gyroscope angular speed to quaternion state update
            gyro = np.array([frame.gx, frame.gy, frame.gz]).astype(np.float32)
            gyro_norm = float(norm(gyro))
            q_gyro = np.array([1, 0, 0, 0]).astype(np.float32)
            if gyro_norm > 0.0000001:
                q_gyro = np.concatenate((np.array([cos(gyro_norm * dt * 0.5)]),
                                         gyro / gyro_norm * sin(gyro_norm * dt * 0.5)), axis=0)
            if round(norm(q_gyro), 8) != 0:
                q_gyro /= norm(q_gyro)

            self.q_gyro_state = quaternion_multiply(self.q_gyro_state, q_gyro)  # (211)

            # 3. Linear interpolation
            # 3.1 get the angles
            [gyro_yaw, gyro_pitch, gyro_roll] = quat_to_ypr(self.q_gyro_state)
            [accel_yaw, accel_pitch, accel_roll] = quat_to_ypr(q_acc)
            # 3.2 complementary filter considering that the gyro yaw is correct
            [yaw, pitch, roll] = np.array([gyro_yaw, gyro_pitch, gyro_roll]) * (1 - alpha) + \
                                 np.array([gyro_yaw, accel_pitch, accel_roll]) * alpha
            # 3.3 Save to quaternion and update state
            self.q_gyro_state = ypr_to_quat(ypr=[yaw, pitch, roll])

            # Pygames visualization
            if self.visualize:
                self.cam(self.q_gyro_state, name=f"q_update, time = {frame.ts / 1000}, yaw = {yaw}")
                # self.cam(np.array([yaw, pitch, roll]) / DEGREE_TO_RAD, name=f"q_update, time = {frame.ts/1000}, dt = {dt}", useQuat=False)
                # in case of not passing quaternion: [yaw, pitch, roll] in DEGREES

            # 4. Express the gravity vector in the local frame
            # To recreate [0, 0, -g], apply this formula:
            local_gravity = quaternion_apply(self.q_gyro_state, [0, 0, -1]) * 9.81

            # Update before returning
            self.last_frame = frame

            return IMUFrame(  # Need to compute that
                ax=frame.ax - local_gravity[0],
                ay=frame.ay - local_gravity[1],
                az=frame.az - local_gravity[2],
                gx=frame.gx,
                gy=frame.gy,
                gz=frame.gz,
                quaternion=self.q_gyro_state,
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

        return np.array(q_vector)


def nlerp(v0, v1, t_):
    '''
    :param v0: first quaternion as np.array
    :param v1: second quaternion as np.array
    :param t_: interpolation coefficient 0: v0, 1: v1
    lerp([1,0,0,0], [0,0,0,1], 0.2)
    :return: interpolated and normalised quaternion
    '''
    if 0 <= t_ <= 1:
        q = (1 - t_) * v0.astype(np.float32) + t_ * v1.astype(np.float32)
    return q / norm(q)


def slerp(v0, v1, t_=0):
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

    theta_0 = acos(dot)
    sin_theta_0 = sin(theta_0)

    theta = theta_0 * t_
    sin_theta = sin(theta)

    s0 = cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * v0) + (s1 * v1)


def quat_to_ypr(q):
    # Output in RAD
    yaw = atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
    pitch = -sin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll = atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
    return np.array([yaw, pitch, roll])


def ypr_to_quat(ypr):  # yaw (Z), pitch (Y), roll (X)
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

    return q / norm(q)


def quat_to_rotMat(quat):
    '''
    :param q: np array representing the quaternion [w, x, y, z]
    :return: the rotation matrix
    '''
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = np.empty((3, 3))

    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)

    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = - x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)

    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = - x2 - y2 + z2 + w2

    check_correct_rot_mat(matrix)
    return matrix


def rotMat_to_quaternion(C):
    '''
    :param rot: rotation matrix
    :return: corresponding quaternion [w, x, y, z]
    '''
    q = 0.5 * np.array([sqrt((1 + np.trace(C))),
                        np.sign(C[2, 1] - C[1, 2]) * sqrt(abs(C[0, 0] - C[1, 1] - C[2, 2] + 1)),
                        np.sign(C[0, 2] - C[2, 0]) * sqrt(abs(C[1, 1] - C[2, 2] - C[0, 0] + 1)),
                        np.sign(C[1, 0] - C[0, 1]) * sqrt(abs(C[2, 2] - C[0, 0] - C[1, 1] + 1))])
    return q / norm(q)

def skewRot(rot):
    # input a matrix, output a matrix
    skew = 0.5 * (rot - rot.T)
    return skew

def skewMatrix(q_n):
    # input a vector, output a matrix
    skew = np.array([[0, -q_n[2], q_n[1]],
                     [q_n[2], 0, -q_n[0]],
                     [-q_n[1], q_n[0], 0]])
    return skew


def rotMat_to_anlgeAxis(rot_mat):
    '''
    :param rot_mat: a rotation matrix
    :return: the rotational vector which describes the rotation as np.array
    '''
    th = acos(0.5 * (rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2] - 1)).real

    if (abs(th) < 0.00000000000001):  # prevent division by 0 in 1 / (2 * sin(th))
        n = np.zeros(3)
    else:
        n = 1 / (2 * sin(th)) * np.array([rot_mat[2, 1] - rot_mat[1, 2],
                                          rot_mat[0, 2] - rot_mat[2, 0],
                                          rot_mat[1, 0] - rot_mat[0, 1]])
    return th * n


def angleAxis_to_quaternion(angleAxis):
    if angleAxis.shape == (3,):
        norm = np.linalg.norm(angleAxis)

        scale = 0
        if norm <= 1e-3:  # small angles
            scale = (0.5 - norm ** 2 / 48 + norm ** 4 / 3840)
        else:
            scale = (np.sin(norm / 2) / norm)

        quat = np.array([1., 0., 0., 0.])
        quat[0] = np.cos(norm / 2)
        quat[1:] = scale * angleAxis

        return quat / np.linalg.norm(quat)
    else:
        sys.exit(f"angle axis input shape {angleAxis.shape} instead of (3,)")

def quaternion_to_angleAxis(q):
    if q.shape == (4,):
        q = q / norm(q)
        # # Rotation library, NOT working properly
        return rotMat_to_anlgeAxis(quaternion_to_rotMat(q))
    else:
        sys.exit(f"angle axis input shape {q.shape} instead of (4,)")

def quaternion_to_rotMat(q):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    sqw = w ** 2
    sqx = x ** 2
    sqy = y ** 2
    sqz = z ** 2

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = (sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = x * y
    tmp2 = z * w
    m10 = 2.0 * (tmp1 + tmp2) * invs
    m01 = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = x * z
    tmp2 = y * w
    m20 = 2.0 * (tmp1 - tmp2) * invs
    m02 = 2.0 * (tmp1 + tmp2) * invs
    tmp1 = y * z
    tmp2 = x * w
    m21 = 2.0 * (tmp1 + tmp2) * invs
    m12 = 2.0 * (tmp1 - tmp2) * invs

    rot = np.array([[m00, m01, m02],
                    [m10, m11, m12],
                    [m20, m21, m22]])

    return rot


def angleAxis_to_rotMat(angleAxis):  # bad for small angles
    th = norm(angleAxis)
    if th < 1e-10:
        return np.eye(1)
    n = angleAxis / th

    i = cos(th) * np.eye(3)
    j = sin(th) * skewMatrix(n)
    k = (1 - cos(th)) * n.dot(n.T)
    return i + j + k
    # return quat_to_rotMat(angleAxis_to_quaternion(angleAxis))


def rotMat_to_ypr(rot):
    q = rotMat_to_quaternion(rot)
    return quat_to_ypr(q)  # [yaw, pitch, roll]


# Quaternion kinematics for the error-state Kalman Filter, Joan Sola, November 8, 2017
def quaternion_multiply(quaternion0, quaternion1):
    # DO NOT Normalise, this would result in wrong calculations when rotating a vector
    [w0, x0, y0, z0] = quaternion0
    [w1, x1, y1, z1] = quaternion1

    output = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                       x0 * w1 + w0 * x1 - z0 * y1 + y0 * z1,
                       y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1,
                       z0 * w1 + w0 * z1 - y0 * x1 + x0 * y1],
                      dtype=np.float64)
    return output


def quaternion_apply(quaternion: List, vector: List):
    # DO NOT normalise the vector
    q2 = np.concatenate((np.array([0.0]), np.array(vector)))
    quaternion = quaternion / norm(quaternion)
    vector = quaternion_multiply(quaternion_multiply(quaternion, q2), quaternion_conjugate(quaternion))[1:]
    return vector


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
def visualize_distance_metric(best_match, degrees, imu_angles, vo_angles) -> None:
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
    plt.scatter(visualize_distance_metric.counter, best_match[1])
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

import collections
from math import pi, atan, sqrt, sin, cos, tan, acos
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import norm

#from .cam import CameraPygame

from math import atan2, sqrt, cos, sin

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])
VOResult = collections.namedtuple("VOResult", ["homogs", "pairs", "ts0", "ts1", "image"])

DEG_TO_RAD = pi / 180.0
RAD_TO_DEG = 180.0 / pi
LERP_THRESHOLD = 0.9
G_ACCEL = 9.80600


def degree_to_rad(angle: float) -> float:
    return angle * DEG_TO_RAD


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
    # https://www.mdpi.com/1424-8220/15/8/19302/htm
    def __init__(self, alpha: float = 0.4):
        self.last_frame = None
        self.current_frame = None
        self.pose = Pose()
        self.alpha = alpha
        #self.cam = CameraPygame()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Orientation of global frame with respect to local frame
        self.q_g_l = np.array([0.0, 1.0, 0.0, 0.0])

    def __call__(self, frame: IMUFrame) -> IMUFrame:
        if self.last_frame is None:
            self.last_frame = frame
            self.current_frame = frame
            return IMUFrame(
                ax=frame.ax,
                ay=frame.ay,
                az=frame.az,
                gx=frame.gx,
                gy=frame.gy,
                gz=frame.gz,
                ts=frame.ts)
        else:
            self.last_frame = self.current_frame
            self.current_frame = frame
            dt_s = (self.last_frame.ts - self.current_frame.ts)/1000.0

            # Normalized accelerations
            a_l = -1.0*np.array([frame.ax, frame.ay, frame.az])
            a_l /= np.linalg.norm(a_l)

            # PREDICTION
            w_q_l = np.array([0.0, frame.gx, frame.gy, frame.gz])*DEG_TO_RAD
            w_q_l /= np.linalg.norm(w_q_l)

            # Gyro based attitude velocity of global frame with respect to local frame
            q_w_dot_g_l = quaternion_multiply_alt(-0.5*w_q_l, self.q_g_l)

            # Gyro based attitude
            q_w_g_l = self.q_g_l + q_w_dot_g_l * dt_s
            q_w_g_l /= np.linalg.norm(q_w_g_l)

            # Inverse gyro based attitude
            q_w_l_g = quaternion_conjugate(q_w_g_l)

            # CORRECTION
            R_q_w_l_g = quaternion_R(q_w_l_g)

            # Compute a prediction for the gravity vector
            g_predicted_g = np.dot(R_q_w_l_g, a_l)

            # Compute delta q acc
            gx, gy, gz = g_predicted_g
            gz_1 = gz + 1.0
            delta_q_acc = np.array([sqrt(gz_1/2.0), -gy/sqrt(2.0*gz_1), gx/sqrt(2.0*gz_1), 0.0])
            delta_q_acc_norm = np.linalg.norm(delta_q_acc)

            delta_q_acc /= delta_q_acc_norm

            q_identity = np.array([1.0, 0.0, 0.0, 0.0])

            # Omega is given by angle subtended by the two quaternions, in our case just:
            omega = delta_q_acc[0]

            if omega > LERP_THRESHOLD:
                delta_q_acc_hat = (1.0 - self.alpha)*q_identity + self.alpha*delta_q_acc
            else:
                delta_q_acc_hat = sin((1.0 - self.alpha)*omega)/sin(omega)*q_identity + sin(self.alpha*omega)/sin(omega)*delta_q_acc

            delta_q_acc_hat_norm = np.linalg.norm(delta_q_acc_hat)
            delta_q_acc_hat /= delta_q_acc_hat_norm

            # UPDATE
            self.q_g_l = quaternion_multiply_alt(q_w_g_l, delta_q_acc_hat)

            # Pygames visualization
            #self.cam(self.q_g_l, name="q_update")

            local_gravity = quaternion_apply(self.q_g_l, [0, 0, -1])[1:] * G_ACCEL

            global_att = quaternion_to_euler(self.q_g_l, False)
            self.roll = global_att[0]
            self.pitch = global_att[1]
            self.yaw = global_att[2]

            return IMUFrame(  # Need to compute that
                ax=frame.ax - local_gravity[0],
                ay=frame.ay - local_gravity[1],
                az=frame.az - local_gravity[2],
                gx=frame.gx,
                gy=frame.gy,
                gz=frame.gz,
                ts=frame.ts
            )

# Marco: Quaternion multiply according to Valenti, 2015
def quaternion_multiply_alt(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    output = np.array([p0*q0 - p1*q1 - p2*q2 - p3*q3,
                       p0*q1 + p1*q0 + p2*q3 - p3*q2,
                       p0*q2 - p1*q3 + p2*q0 + p3*q1,
                       p0*q3 + p1*q2 - p2*q1 + p3*q0], dtype=np.float64)
    output /= np.linalg.norm(output)
    return output


def quaternion_R(q):
    q0, q1, q2, q3 = q
    R = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2.0*(q1*q2 - q0*q3), 2.0*(q1*q3 + q0*q2)],
                  [2.0*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2.0*(q2*q3 - q0*q1)],
                  [2.0*(q1*q3 - q0*q2), 2.0*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
    return R


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

def quaternion_to_euler(quaternion, degrees=True):
    return Rotation.from_quat(quaternion).as_euler('zyx', degrees=degrees)

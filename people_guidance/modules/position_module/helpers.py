import collections
from math import pi, atan, sqrt, sin, cos, tan
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np

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
            if len(self.keys[key]) > window_size:
                self.keys[key].pop(0)

            self.keys[key].append(value)

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
        else:
            dt = self.last_frame.ts - frame.ts
            # complemetary filtering

            pitch_accel = atan(frame.ay / sqrt(frame.az ** 2 + frame.ax ** 2))
            roll_accel = atan(frame.az / sqrt(frame.ay ** 2 + frame.ax ** 2))

            # Pitch, roll and yaw based on gyro
            roll_gyro = frame.gz + \
                        frame.gy * sin(self.pose.pitch) * tan(self.pose.roll) + \
                        frame.gx * cos(self.pose.pitch) * tan(self.pose.roll)

            pitch_gyro = frame.gy * cos(self.pose.pitch) - frame.gx * sin(self.pose.pitch)

            yaw_gyro = frame.gy * sin(self.pose.pitch) * 1.0 / cos(self.pose.roll) + frame.gx * cos(
                self.pose.pitch) * 1.0 / cos(self.pose.roll)

            # Apply complementary filter
            self.pose.pitch = (1.0 - self.alpha) * (self.pose.pitch + pitch_gyro * dt) + self.alpha * pitch_accel
            self.pose.roll = (1.0 - self.alpha) * (self.pose.roll + roll_gyro * dt) + self.alpha * roll_accel
            self.pose.yaw += yaw_gyro * dt

        # TODO: Use this new pose estimate to remove gravity acc from frame, return the frame.
        return IMUFrame()


def visualize_input_data(frames: List[IMUFrame]) -> None:
    #PLOT
    plt.figure(2, figsize=(10, 12))
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


def visualize_distance_metric(best_match, best_match2, degrees, imu_angles, vo_angles, counter) -> int:
    plt.figure(1, figsize=(10, 12))
    plt.tight_layout()

    plt.suptitle(f"Distance evaluation method", x=0.5, y=.999)

    plt.subplot(2, 1, 1)
    plt.scatter(counter, best_match[1])
    plt.title('Distance Theo')
    plt.xlabel('')
    plt.ylabel('')

    plt.subplot(2, 1, 2)
    plt.scatter(counter, best_match2[1])
    plt.title('Distance Adrian')
    plt.xlabel('')
    plt.ylabel('')

    counter += 1
    plt.pause(0.001)

    #PLOT
    plt.figure(1, figsize=(10, 12))
    plt.tight_layout()

    plt.suptitle(f"Evolution of the rotations, degrees? {degrees}", x=0.5, y=.999)

    plt.subplot(2, 1, 1)
    plt.scatter(counter, imu_angles[2])
    plt.title('rot IMU')
    plt.xlabel('')
    plt.ylabel('')

    plt.subplot(2, 1, 2)
    plt.scatter(counter, vo_angles[2])
    plt.title('rot VO')
    plt.xlabel('')
    plt.ylabel('')

    counter += 1
    plt.pause(0.001)
    return counter
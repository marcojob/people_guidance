import collections
from math import pi
from typing import List

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
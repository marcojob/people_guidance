import collections

import numpy as np
from scipy.spatial.transform import Rotation

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])

VOResult = collections.namedtuple("VOResult", ["homogs", "pairs", "ts0", "ts1", "image"])


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

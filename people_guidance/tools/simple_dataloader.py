import collections
from pathlib import Path
import numpy as np

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "quaternion", "ts"])


class SimpleIMUDatalaoder:
    def __init__(self, data_dir):
        self.data_dir: Path = data_dir
        self.lines = []
        self.elem_name_map = {"accel_x": "ax", "accel_y": "ay", "accel_z": "az",
                              "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz"}

    def __enter__(self):
        with open(str(self.data_dir / "imu_data.txt"), "r") as imu_fp:
            self.lines = imu_fp.readlines()

    def __len__(self):
        return len(self.lines)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getitem__(self, i):
        # unit acc: m/s -- unit gyro : deg/s, ---- unit ts: ms
        frame_kwargs = {}

        line = self.lines[i]

        line = line.replace("\n", "").replace("\r", "")
        ts_str, data_str = line.split(":", maxsplit=1)
        ts: int = int(ts_str)
        data_elem_str = data_str.split(", ")
        for elem in data_elem_str:
            label, val_str = elem.replace(" ", "").split(":")
            val = float(val_str)
            frame_kwargs[self.elem_name_map[label]] = val

        frame_kwargs["ts"] = ts
        frame_kwargs["quaternion"] = np.array([1, 0, 0, 0])

        return IMUFrame(**frame_kwargs)
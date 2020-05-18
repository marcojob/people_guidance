from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image


class PositionGenius:
    def __init__(self, dataset_name, dataset_dir):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        self.ground_truth = []

        with open(self.dataset_dir / "groundtruth.txt") as fp:
            next(fp)  # skip the heading
            for line in fp:
                line = line.replace("\n", "")
                #  timestamp tx ty tz qx qy qz qw
                timestamp_str, *position_str = line.split(" ")
                timestamp = int(float(timestamp_str) * 1000)  # convert to ms

                position = tuple((float(param) for param in position_str))
                assert len(position) == 7
                self.ground_truth.append((timestamp, position))

    def __call__(self, timestamp):

        for i, (ts, pos) in enumerate(self.ground_truth):
            if ts == timestamp:
                return pos
            elif ts > timestamp:

                ts_prev, pos_prev = self.ground_truth[i-1]
                position = self.interpolate_position(pos_prev, pos, ts_prev, ts, timestamp)
                return position

    @staticmethod
    def interpolate_position(pos0: tuple, pos1: tuple, ts0: int, ts1: int, timestamp) -> np.array:
        assert ts0 < timestamp < ts1, f"target timestamp {timestamp} for interpolation must lie " \
                                      f"between image timestamps {ts0, ts1}"
        pos_delta = tuple([pos1[i] - pos0[i] for i in len(pos0)])
        lever = (timestamp - ts0) / (ts1 - ts0)
        return pos0 + pos_delta * lever

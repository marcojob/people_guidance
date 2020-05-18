from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np
import cv2
from PIL import Image


class PositionGenius:
    def __init__(self, dataset_dir):
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

                self.ground_truth.sort(key=lambda tup: tup[0])

    def __call__(self, timestamp):
        if timestamp < self.ground_truth[0][0]:
            logging.critical(f"NO ground truth for timestamp {timestamp}")
            return None

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
        pos0np = np.array(pos0)
        pos1np = np.array(pos1)
        pos_delta = pos1np - pos0np
        lever = (timestamp - ts0) / (ts1 - ts0)
        return pos0np + pos_delta * lever

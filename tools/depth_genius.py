from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image


class DepthGenius:
    def __init__(self, dataset_name, dataset_dir, resize_size=None):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        self.truth_timestamps = []
        self.resize_size: Optional[Tuple[int, int]] = resize_size
        self.truth_fpaths = []
        with open(self.dataset_dir / "depth.txt") as fp:
            # 11873.252219439 depth/11873.252219.png
            for line in fp:
                line = line.replace("\n", "")
                timestamp_str, fpath = line.split(" ")
                timestamp = int(float(timestamp_str) * 1000)  # convert to ms

                self.truth_timestamps.append(timestamp)
                self.truth_fpaths.append(fpath)

        assert len(self.truth_fpaths) == len(self.truth_timestamps)

    def __call__(self, points: np.array, timestamp):
        assert len(points.shape) == 2, points.shape[1] == 2
        for i, ts in enumerate(self.truth_timestamps):
            if ts == timestamp:
                dmap = self.load_depth_from_fpath(self.truth_fpaths[i])
                return dmap[points[:, 0], points[:, 1]]
            elif ts > timestamp:
                dmap0 = self.load_depth_from_fpath(self.truth_fpaths[i-1])
                dmap1 = self.load_depth_from_fpath(self.truth_fpaths[i])

                #  print(f"interpolating for timestamp {timestamp} found neighbors at {self.truth_timestamps[i-1]}"
                #      f"and {self.truth_timestamps[i]}")

                dmap = self.interpolate_depth(dmap0, dmap1, self.truth_timestamps[i-1],
                                              self.truth_timestamps[i], timestamp)
                return dmap[points[:, 0], points[:, 1]]

    def load_depth_from_fpath(self, fpath: str):
        arr = np.array(Image.open(self.dataset_dir / fpath))
        if self.resize_size is not None:
            arr = cv2.resize(arr, self.resize_size)
        return arr

    @staticmethod
    def interpolate_depth(dmap0: np.array, dmap1: np.array, ts0, ts1, timestamp) -> np.array:
        assert ts0 < timestamp < ts1, "target timestamp for interpolation must lie between image timestamps"
        depth_delta = dmap1 - dmap0
        lever = (timestamp - ts0) / (ts1 - ts0)
        return dmap0 + depth_delta * lever


if __name__ == '__main__':
    dg = DepthGenius("cables_2", Path("converted_eth_slam_cables_2"))
    points = np.array(((12, 13), (200, 200), (100, 13)))
    print(points.shape)
    true_depth = dg(points, 11873357)
    print(true_depth)
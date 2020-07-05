from typing import Dict, List, Union

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm

from people_guidance.tools.simple_dataloader import SimpleIMUDatalaoder
from people_guidance.tools.position_genius import PositionGenius
from people_guidance.utils import ROOT_DATA_DIR
from people_guidance.tools.comp_filter import ComplementaryFilter


class Position:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.ang_x = 0.0  # in degrees
        self.ang_y = 0.0  # in degrees
        self.ang_z = 0.0  # in degrees


class ValueList3D:
    def __init__(self, name=None):
        self.name = name
        self.x = []
        self.y = []
        self.z = []

    def append(self, x, y, z):
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)


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


if __name__ == '__main__':

    dataset_dir = ROOT_DATA_DIR / "converted_eth_slam_large_loop_1"
    dataloader = SimpleIMUDatalaoder(dataset_dir)
    pg = PositionGenius(dataset_dir)
    avg_filter = MovingAverageFilter()
    cpl_pos = Position()
    direct_pos = Position()
    complementary_filter = ComplementaryFilter()

    last_frame = None
    true_pos_offset = None
    direct_pred = ValueList3D()
    compl_pred = ValueList3D()
    true_pred = ValueList3D()

    timestamps = []
    dts = []

    with dataloader:
        for frame in tqdm(dataloader):
            true_position = pg(frame.ts)
            if last_frame is not None and true_position is not None:
                # ignore first frame and frames for which we dont have ground truth
                dt = (frame.ts - last_frame.ts) / 1000.0  # convert to seconds
                dts.append(dt)

                cpl_pos.ang_x, cpl_pos.ang_y, cpl_pos.ang_z = complementary_filter(frame, dt)
                compl_pred.append(cpl_pos.ang_x, cpl_pos.ang_y, cpl_pos.ang_z)

                direct_pos.ang_x += dt * avg_filter("gx", frame.gx, window_size=400)
                direct_pos.ang_y += dt * avg_filter("gy", frame.gy, window_size=400)
                direct_pos.ang_z += dt * avg_filter("gz", frame.gz, window_size=400)
                direct_pred.append(direct_pos.ang_x,  direct_pos.ang_y, direct_pos.ang_z)

                true_pred = Rotation.from_quat(true_position[:4]).as_euler("xyz", degrees=True)
                if true_pos_offset is None:
                    # we want to move the origin to be equal to the first true pose
                    # since we start the IMU integration also at 0.
                    true_pos_offset = true_pred

                true_pred.append(true_pred[0] - true_pos_offset[0],
                                 true_pred[1] - true_pos_offset[1],
                                 true_pred[2] - true_pos_offset[2])

                timestamps.append(frame.ts)

            last_frame = frame

    # plt.plot(timestamps, dts, c="y")
    plt.plot(timestamps, true_pred.x, c="g")
    plt.plot(timestamps, compl_pred.x, c="y")
    plt.plot(timestamps, direct_pred.x, c="r")
    plt.show()
from time import sleep
from typing import Dict, List, Union

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from people_guidance.tools.simple_dataloader import SimpleIMUDatalaoder
from people_guidance.tools.position_genius import PositionGenius
from people_guidance.utils import ROOT_DATA_DIR
# from people_guidance.tools.comp_filter import
from people_guidance.modules.position_module.helpers import ComplementaryFilter, \
    MovingAverageFilter, pygameVisualize, IMUFrame
from people_guidance.modules.position_module.helpers import quaternion_apply, quaternion_conjugate


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

if __name__ == '__main__':

    dataset_dir = ROOT_DATA_DIR / "converted_eth_slam_large_loop_1"
    dataloader = SimpleIMUDatalaoder(dataset_dir)
    pg = PositionGenius(dataset_dir)
    avg_filter = MovingAverageFilter()
    cpl_pos = Position()
    direct_pos = Position()
    complementary_filter = ComplementaryFilter()

    vispg = pygameVisualize()

    last_frame = None
    true_pos_offset = None
    direct_pred = ValueList3D()
    compl_pred = ValueList3D()
    true_pred = ValueList3D()

    timestamps = []
    dts = []

    quat_pg_list = []
    imu_frame_list: List[IMUFrame] = []

    counter = 0

    with dataloader:
        for frame in tqdm(dataloader):
            # print("IMU sandbox, frame:", frame)
            true_position = pg(frame.ts)
            # print("true position", true_position)

            # Limit the length of the dataset
            if counter > 500:
                break

            if last_frame is not None and true_position is not None:
                # ignore first frame and frames for which we dont have ground truth
                # print("original frame", frame)
                # print("true position", true_position)

                # Limit the length of the dataset
                counter += 1

                # timestamp
                timestamps.append(frame.ts)

                # accel
                accel_frame = np.array([frame.ax, frame.ay, frame.az])
                print("accel_frame", accel_frame)
                # print("rebuilt accel quat*g", quaternion_apply(quaternion_conjugate(quat_pg), [0, 0, -9.8])) # DOES NOT give the accel_frame back even if the motion is ~0

                # # dt
                # dt = (frame.ts - last_frame.ts) / 1000.0  # convert to seconds
                # dts.append(dt)

                # complementary filter
                imu_frame = complementary_filter(frame, alpha=0.9)  # alpha = 0 : gyro, alpha = 1 : accel
                imu_frame_list.append(imu_frame)
                # print('compl filter output', imu_frame)
                # print("IMU quaternion", imu_frame.quaternion)

                # quaternion ground truth
                quat_pg = np.array([true_position[6], true_position[3], true_position[4], true_position[5]])
                quat_pg_list.append(quat_pg)
                # print("quat_pg", quat_pg)
                quat____ = np.array([true_position[3], true_position[4], true_position[5], true_position[6]])
                # print("rebuild g from quat*accel ROTATION", Rotation.from_quat(quat____).apply(accel_frame))

                # accel IMU
                print("rebuilt g from quat_IMU*accel", quaternion_apply(imu_frame.quaternion, accel_frame))


            #     cpl_pos.ang_x, cpl_pos.ang_y, cpl_pos.ang_z = complementary_filter(frame, dt)
            #     compl_pred.append(cpl_pos.ang_x, cpl_pos.ang_y, cpl_pos.ang_z)
            #
            #     direct_pos.ang_x += dt * avg_filter("gx", frame.gx, window_size=400)
            #     direct_pos.ang_y += dt * avg_filter("gy", frame.gy, window_size=400)
            #     direct_pos.ang_z += dt * avg_filter("gz", frame.gz, window_size=400)
            #     direct_pred.append(direct_pos.ang_x,  direct_pos.ang_y, direct_pos.ang_z)
            #
            #     true_pred = Rotation.from_quat(true_position[:4]).as_euler("xyz", degrees=True)
            #     if true_pos_offset is None:
            #         # we want to move the origin to be equal to the first true pose
            #         # since we start the IMU integration also at 0.
            #         true_pos_offset = true_pred
            #
            #     true_pred.append(true_pred[0] - true_pos_offset[0],
            #                      true_pred[1] - true_pos_offset[1],
            #                      true_pred[2] - true_pos_offset[2])
            #
            #     timestamps.append(frame.ts)

            last_frame = frame

    counter = 0
    for quat_pg, quat_compl, time in zip(quat_pg_list, imu_frame_list, timestamps):
        if counter > 10:
            counter = 0
            # vispg(quat_pg, visualize=True, name=f"TRUE quat, ts = {time}", rmv_yaw=True)
            # sleep(0.02)
            vispg(quat_compl.quaternion, visualize=True, name=f"Compl imu quat, ts = {time}", rmv_yaw=False)
            sleep(0.02)
        counter += 1

    # # plt.plot(timestamps, dts, c="y")
    # plt.plot(timestamps, true_pred.x, c="g")
    # plt.plot(timestamps, compl_pred.x, c="y")
    # plt.plot(timestamps, direct_pred.x, c="r")
    # plt.show()
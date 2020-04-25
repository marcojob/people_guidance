import pathlib
import time
from typing import Dict, Tuple, Optional, List, Generator

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from ..module import Module
from .helpers import IMUFrame, VOResult, Homography, interpolate_frames


class PositionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super().__init__(name="position_module",
                         outputs=[("homography", 10)],
                         inputs=["drivers_module:accelerations",
                                 "feature_tracking_module:feature_point_pairs"],
                         log_dir=log_dir)

        self.vo_buffer = []
        self.imu_buffer = []

    def start(self):
        while True:
            self.get_inputs()
            self.prune_buffers()
            self.predict_relative_pose()

    def get_inputs(self):

        vo_payload: Dict = self.get("feature_tracking_module:feature_point_pairs")
        if vo_payload:
            self.vo_buffer.append(self.vo_result_from_payload(vo_payload))
        if len(self.imu_buffer) < 100:
            imu_payload: Dict = self.get("drivers_module:accelerations")
            if imu_payload:
                self.imu_buffer.append(self.imu_frame_from_payload(imu_payload))
            else:
                time.sleep(0.001)

    @staticmethod
    def imu_frame_from_payload(payload: Dict) -> IMUFrame:
        return IMUFrame(
            ax=float(payload['data']['accel_x']),
            ay=float(payload['data']['accel_y']),
            az=float(payload['data']['accel_z']),
            gx=float(payload['data']['gyro_x']),
            gy=float(payload['data']['gyro_y']),
            gz=float(payload['data']['gyro_z']),
            ts=payload['data']['timestamp']
        )

    # imu 10, 11, 12, 13 ,14

    # vo 13, 14, 15

    @staticmethod
    def vo_result_from_payload(payload: Dict):
        return VOResult(homogs=payload["data"]["camera_positions"], pairs=payload["data"]["point_pairs"],
                        ts0=payload["data"]["timestamp_pair"][0], ts1=payload["data"]["timestamp_pair"][1],
                        image=payload["data"]["image"])

    def prune_buffers(self):
        if len(self.vo_buffer) > 1 and len(self.imu_buffer) > 1:
            self.prune_vo_buffer()
        if len(self.vo_buffer) > 1 and len(self.imu_buffer) > 1:
            self.prune_imu_buffer()

    def prune_vo_buffer(self):
        # this should rarely happen. It discards items from the vo buffer that have timestamps older than
        # the oldest still buffered imu frame.
        for _ in range(len(self.vo_buffer)):
            if self.vo_buffer[0].ts0 < self.imu_buffer[0].ts:
                self.vo_buffer.pop(0)

    def prune_imu_buffer(self):

        i = 0
        # find the index i of the element that has a larger timestamp than the oldest vo_result still in the buffer.
        while i < len(self.imu_buffer) and self.imu_buffer[i].ts < self.vo_buffer[0].ts0:
            i += 1
        # we assume that the imu_frames are sorted by timestamp: remove the first i-1 elements from the imu_buffer
        for _ in range(i-2):
            self.imu_buffer.pop(0)

    def predict_relative_pose(self):
        prune_idxs = []
        for idx, vo_result in enumerate(self.vo_buffer):
            i0, i1 = self.find_imu_integration_interval(vo_result.ts0, vo_result.ts1)

            if i0 is None:
                # if we cant find imu data that is older than the vo result we want to discard the vo result.
                prune_idxs.append(idx)

            if i0 is not None and i1 is not None:
                # if we have both slightly older and newer imu data than the interval covered by the vo_result we
                # integrate our imu data in that interval.
                self.logger.info(f"Found frames with timestamps: i0 {self.imu_buffer[i0].ts} t0 {vo_result.ts0} ts1 "
                                 f"{vo_result.ts1} i1 {self.imu_buffer[i1].ts}")

                frames: List[IMUFrame] = self.find_integration_frames(vo_result.ts0, vo_result.ts1, i0, i1)
                imu_homography: Homography = self.integrate(frames)
                homog: np.array = self.choose_nearest_homography(vo_result, imu_homography)
                prune_idxs.append(idx)

                self.publish("homography", {"homography": homog, "point_pairs": vo_result.pairs,
                                            "timestamps": (vo_result.ts0, vo_result.ts1),
                                            "image": vo_result.image}, 100)

        for offset, idx in enumerate(prune_idxs):
            # we assume that the prune_idxs are sorted low to high
            self.vo_buffer.pop(idx - offset)

    def find_imu_integration_interval(self, ts0, ts1) -> List[Optional[int]]:
        # returns indices from self.imu_buffer, which form the interval over which we want to integrate
        neighbors = [None, None]

        for idx, frame in enumerate(self.imu_buffer):
            # this assumes that our frames are sorted old to new.
            if frame.ts <= ts0:
                neighbors[0] = idx
            if frame.ts >= ts1:
                neighbors[1] = idx
                break
        return neighbors

    def integrate(self, frames: List[IMUFrame]) -> Homography:
        pos = Homography()

        for i in range(1, len(frames)):
            dt = (frames[i].ts - frames[i-1].ts) / 1000
            dt2 = dt * dt
            pos.x += 0.5 * frames[i].ax * dt2
            pos.y += 0.5 * frames[i].ay * dt2
            pos.z += 0.5 * frames[i].az * dt2
            pos.roll += frames[i].gx * dt
            pos.pitch += frames[i].gy * dt
            pos.yaw += frames[i].gz * dt
            # this is just a placeholder. Here we have to do the proper fusion of gyro/acc data etc.

        return pos

    def find_integration_frames(self, ts0, ts1, i0, i1) -> List[IMUFrame]:
        integration_frames: List[IMUFrame] = []
        lower_frame_bound: IMUFrame = interpolate_frames(self.imu_buffer[i0], self.imu_buffer[i0 + 1], ts0)
        integration_frames.append(lower_frame_bound)

        for j in range(i0+1, i1-1):
            integration_frames.append(self.imu_buffer[j])

        upper_frame_bound: IMUFrame = interpolate_frames(self.imu_buffer[i1 - 1], self.imu_buffer[i1], ts1)
        integration_frames.append(upper_frame_bound)
        return integration_frames

    def choose_nearest_homography(self, vo_result: VOResult, imu_homog: Homography) -> np.array:
        imu_homog_matrix = imu_homog.as_matrix()
        best_match = (None, np.inf)

        for homog in vo_result.homogs:
            k_t = 1
            imu_rot = Rotation.from_matrix(imu_homog_matrix[0:3, 0:3])
            imu_t_vec = imu_homog_matrix[0:3, 3]
            vo_rot = Rotation.from_matrix(homog[0:3, 0:3])
            vo_t_vec = homog[0:3, 3]
            delta_rot: Rotation = imu_rot.inv() * vo_rot
            distance = delta_rot.magnitude() + k_t * np.linalg.norm(imu_t_vec - vo_t_vec)

            if distance < best_match[1]:
                best_match = (homog, distance)

        imu_angles = Rotation.from_matrix(imu_homog_matrix[..., :3]).as_euler('xyz', degrees=True)
        vo_angles = Rotation.from_matrix(best_match[0][..., :3]).as_euler("xyz", degrees=True)
        imu_xyz = str(imu_homog_matrix[..., 3:]).replace("\n", "")
        vo_xyz = str(best_match[0][..., 3:]).replace("\n", "")
        self.logger.info(f"Prediction Offset:\n"
                         f"IMU angles :{imu_angles}\nVO angles  :{vo_angles}\n"
                         f"IMU pos :{imu_xyz}\nVO pos  :{vo_xyz}")

        return best_match[0]

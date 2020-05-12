
import pathlib
import time
from typing import Dict, Tuple, Optional, List, Generator, Union

import numpy as np
from scipy.spatial.transform import Rotation
from math import tan, atan2, cos, sin, pi, sqrt, atan, acos

from ..module import Module
from .helpers import IMUFrame, VOResult, Homography, interpolate_frames, visualize_input_data, \
    visualize_distance_metric
from .helpers import degree_to_rad, MovingAverageFilter, ComplementaryFilter, Velocity
from .helpers import rotMat_to_anlgeAxis, quat_to_rotMat, rotMat_to_ypr
from .helpers import check_correct_rot_mat, normalise_rotation


class PositionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super().__init__(name="position_module",
                         outputs=[("homography", 10)],
                         inputs=["drivers_module:accelerations",
                                 "feature_tracking_module:feature_point_pairs"],
                         log_dir=log_dir)

        self.vo_buffer: List[VOResult] = []
        self.imu_buffer: List[IMUFrame] = []

        self.avg_filter = MovingAverageFilter()
        self.complementary_filter = ComplementaryFilter()

        self.velocity = Velocity()

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

    def imu_frame_from_payload(self, payload: Dict) -> IMUFrame:
        # In Camera coordinates: X = -Z_IMU, Y = Y_IMU, Z = X_IMU (90° rotation around the Y axis)
        frame = IMUFrame(
            ax=self.avg_filter("ax", -float(payload['data']['accel_z']), 10), # m/s ** 2
            ay=self.avg_filter("ay", float(payload['data']['accel_y']), 10),
            az=self.avg_filter("az", float(payload['data']['accel_x']), 10),
            gx=self.avg_filter("gx", -degree_to_rad(float(payload['data']['gyro_z'])), 10), # input: °/s, output : RAD/s
            gy=self.avg_filter("gy", degree_to_rad(float(payload['data']['gyro_y'])), 10),
            gz=self.avg_filter("gz", degree_to_rad(float(payload['data']['gyro_x'])), 10),
            quaternion=[1, 0, 0, 0],
            ts=payload['data']['timestamp']
        )
        self.logger.info(f"Input frame from driver : \n{frame}")

        # Combine Gyro and Accelerometer data to extract the gravity and add the current rotation to *frame*
        return self.complementary_filter(frame, alpha=0.5)


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
                # self.logger.info(f"IMU : {imu_homography.roll}, {imu_homography.pitch}, {imu_homography.yaw}")

                homog: np.array = self.choose_nearest_homography(vo_result, imu_homography)
                prune_idxs.append(idx)

                self.publish("homography", {"homography": homog, "point_pairs": vo_result.pairs,
                                            "timestamps": (vo_result.ts0, vo_result.ts1),
                                            "image": vo_result.image}, 100)

        for offset, idx in enumerate(prune_idxs):
            # we assume that the prune_idxs are sorted low to high
            self.vo_buffer.pop(idx - offset)

    def find_imu_integration_interval(self, ts0, ts1) -> List[Optional[int]]:
        # returns indices from self.imu_buffer, which forms the interval over which we want to integrate
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

        #PLOT
        # visualize_input_data(frames)

        # Displacement
        # TODO rotate the frames before integrating, write the update in original coordinates
        for i in range(1, len(frames)):
            dt = (frames[i].ts - frames[i-1].ts) / 1000
            dt2 = dt * dt

            pos.x += self.velocity.x * dt + 0.5 * frames[i].ax * dt2
            pos.y += self.velocity.y * dt + 0.5 * frames[i].ay * dt2
            pos.z += self.velocity.z * dt + 0.5 * frames[i].az * dt2

            self.velocity.x = (self.velocity.x + frames[i].ax * dt)
            self.velocity.y = (self.velocity.y + frames[i].ay * dt)
            self.velocity.z = (self.velocity.z + frames[i].az * dt)

            self.velocity.dampen()

            self.logger.debug(f"pos calculated {pos}")

        # Rotation between the first and last frame
        rot_first = quat_to_rotMat(frames[0].quaternion)
        rot_last = quat_to_rotMat(frames[-1].quaternion)
        # Difference in rotation
        pos.rotation_matrix = rot_first.T.dot(rot_last)
        # Extraction of the angle axis from the rotation matrix
        [pos.roll, pos.pitch, pos.yaw] = rotMat_to_anlgeAxis(pos.rotation_matrix)

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
        imu_homog_matrix = imu_homog.as_Tmatrix()
        imu_rot = imu_homog_matrix[0:3, 0:3]
        imu_t_vec = imu_homog_matrix[0:3, 3]

        best_match = (None, None, np.inf, None)

        for homog in vo_result.homogs:
            k_t = 0.0

            # Correction: Homography gives a result rotated from our camera coordinate frame
            vo_to_camera = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]])
            vo_rot = homog[0:3, 0:3]
            vo_rot = normalise_rotation(vo_rot)
            check_correct_rot_mat(vo_rot)

            vo_t_vec = vo_to_camera.dot(homog[0:3, 3])

            # Robot dynamic script p51
            # Issues for rotation angles close to zero or pi
            delta_rot = rotMat_to_anlgeAxis(imu_rot.T.dot(vo_rot))
            distance = np.linalg.norm(delta_rot) + k_t * np.linalg.norm(imu_t_vec - vo_t_vec)

            # self.logger.debug(f"VO as euler \n{Rotation.from_matrix(homog[0:3, 0:3]).as_euler('xyz', degrees=True)}")
            # self.logger.debug(f"rotate VO frame by \n{vo_to_camera}")
            # self.logger.debug(f"VO rotated \n{vo_rot.as_euler('xyz', degrees=True)}")
            # self.logger.debug(f"Data from VO: angles: {vo_rot.as_euler('xyz', degrees=True)}, displacement: {vo_t_vec}")

            if distance < best_match[2]:
                best_match = (vo_rot, vo_t_vec, distance, delta_rot)

        imu_ypr = rotMat_to_ypr(imu_homog_matrix[..., :3])
        vo_ypr = rotMat_to_ypr(best_match[0])
        imu_xyz = str(imu_homog_matrix[..., 3:]).replace("\n", "")
        vo_xyz = str(best_match[1]).replace("\n", "")
        self.logger.warning(f"Prediction Offset:\n"
                         f"IMU Euler [yaw, pitch, roll] angles :\n{imu_ypr}\nVO angles :\n{vo_ypr}, (RAD)\n"
                         f"IMU pos :{imu_xyz}\nVO pos  :{vo_xyz}")

        #PLOT
        # visualize_distance_metric(best_match, best_match2, degrees, imu_angles, vo_angles)

        return np.column_stack((best_match[0], best_match[1]))



import pathlib
import time
from typing import Dict, Tuple, Optional, List, Generator, Union

import numpy as np
from scipy.spatial.transform import Rotation
from math import tan, atan2, cos, sin, pi, sqrt, atan, acos

from ..module import Module
from .helpers import IMUFrame, VOResult, Homography, interpolate_frames
from .helpers import visualize_input_data, visualize_distance_metric, pygameVisualize
from .helpers import degree_to_rad, MovingAverageFilter, ComplementaryFilter, Velocity
from .helpers import rotMat_to_anlgeAxis, rotMat_to_ypr, angleAxis_to_rotMat, quaternion_to_rotMat, \
    angleAxis_to_quaternion, rotMat_to_quaternion, quaternion_apply, quat_to_ypr, quaternion_multiply, quaternion_conjugate
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

        self.visualize = 1 # 0 : none, 1: imu compl output, 2: Interpolate function output, 3: integrated rot, 10: best match
        if not self.visualize==0:
            self.vispg = pygameVisualize()

        self.time_tracker = time.perf_counter_ns()

    def start(self):
        while True: # <2ms usually. ~10ms when the relative pose can be predicted
            self.get_inputs() # 1 to 2.5 ms whenever called (BAD)
            self.prune_buffers() # less than 0.1 ms
            self.predict_relative_pose() # < 8 ms when handling data

    def get_inputs(self):
        vo_payload: Dict = self.get("feature_tracking_module:feature_point_pairs")
        if vo_payload:
            self.vo_buffer.append(self.vo_result_from_payload(vo_payload)) # 0.3 up to 0.7 ms to get and append
            print(f"time vo_buffer ts1 input = {self.vo_buffer[-1].ts1 / 1000}")
        if len(self.imu_buffer) < 100:
            imu_payload: Dict = self.get("drivers_module:accelerations")
            if imu_payload:
                # print('IMU input', imu_payload)
                self.imu_buffer.append(self.imu_frame_from_payload(imu_payload)) # about 1ms to execute
                print(f"timestamp IMU input = {self.imu_buffer[-1].ts / 1000}")
                delta_t = time.perf_counter_ns() - self.time_tracker
                self.time_tracker = time.perf_counter_ns()
                print(f"time since last received IMU (ns) {delta_t}")
            else:
                time.sleep(0.001)

    def imu_frame_from_payload(self, payload: Dict) -> IMUFrame:
        # In Camera coordinates: X = -Z_IMU, Y = Y_IMU, Z = X_IMU (90° rotation around the Y axis)
        n_avg = 5
        frame = IMUFrame(
            ax=self.avg_filter("ax", -float(payload['data']['accel_z']), n_avg), # m/s ** 2
            ay=self.avg_filter("ay", float(payload['data']['accel_y']), n_avg),
            az=self.avg_filter("az", float(payload['data']['accel_x']), n_avg),
            gx=self.avg_filter("gx", -degree_to_rad(float(payload['data']['gyro_z'])), n_avg), # input: °/s, output : RAD/s
            gy=self.avg_filter("gy", degree_to_rad(float(payload['data']['gyro_y'])), n_avg),
            gz=self.avg_filter("gz", degree_to_rad(float(payload['data']['gyro_x'])), n_avg),
            quaternion=[1, 0, 0, 0],
            ts=payload['data']['timestamp']
        )
        self.logger.debug(f"Input frame from driver : \n{frame}") # TODO: Problem: gravity ist raus

        # Combine Gyro and Accelerometer data to extract the gravity and add the current rotation to *frame*
        imu_frame = self.complementary_filter(frame, alpha=0.1) # alpha = 0 : gyro, alpha = 1 : accel

        self.logger.debug(f"Frame after complementary filter : \n{imu_frame}")
        if self.visualize == 1:
            self.vispg(imu_frame.quaternion, visualize=True, name="IMU rotation, alpha = 0.1") # amazing

        return imu_frame # correct output


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
        # print("self.vo_buffer", len(self.vo_buffer))
        for idx, vo_result in enumerate(self.vo_buffer):
            i0, i1 = self.find_imu_integration_interval(vo_result.ts0, vo_result.ts1)

            if i0 is None:
                # if we cant find imu data that is older than the vo result we want to discard the vo result.
                prune_idxs.append(idx)
                self.logger.critical('discarding some vo data, no old enough IMU data')

            if i0 is not None and i1 is not None:
                # if we have both slightly older and newer imu data than the interval covered by the vo_result we
                # integrate our imu data in that interval.
                self.logger.debug(f"Found frames with timestamps: i0 {self.imu_buffer[i0].ts} t0 {vo_result.ts0} ts1 "
                                 f"{vo_result.ts1} i1 {self.imu_buffer[i1].ts}")

                frames: List[IMUFrame] = self.find_integration_frames(vo_result.ts0, vo_result.ts1, i0, i1)
                imu_homography: Homography = self.integrate(frames)
                # self.logger.info(f"IMU : {imu_homography.roll}, {imu_homography.pitch}, {imu_homography.yaw}")

                homog: np.array = self.choose_nearest_homography(vo_result, imu_homography)
                prune_idxs.append(idx)

                self.publish("homography", {"homography": homog, "point_pairs": vo_result.pairs,
                                            "timestamps": (vo_result.ts0, vo_result.ts1),
                                            "image": vo_result.image}, 100)

            if i0 is not None and i1 is None:
                # No recent enough IMU data
                self.logger.debug(f"no recent enough IMU data for VO relative pose estimation.\n"
                                    f"Index is {idx} out of {len(self.vo_buffer)}.")
                # Assumes that older data is appended to the end
                break

        for offset, idx in enumerate(prune_idxs):
            # we assume that the prune_idxs are sorted low to high
            print(f"pruning elt {idx - offset} from Vo_buffer of length {len(self.vo_buffer)}")
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

        # Rotation between the first and last frame
        rot_first = quaternion_to_rotMat(frames[0].quaternion) # Rotation is correct
        rot_last = quaternion_to_rotMat(frames[-1].quaternion)

        # TODO was set to quat_to_rotMat (wrong solution)
        # Rotation is wrong: roll and yaw swapped

        # Difference in rotation
        pos.rotation_matrix = rot_first.T.dot(rot_last) # Rotation is wrong: roll and yaw swapped
        quat_update = quaternion_multiply(quaternion_conjugate(frames[0].quaternion), frames[-1].quaternion) # seems correct

        if self.visualize == 5:
            try:
                self.current_rot_test = quaternion_multiply(self.current_rot_test, quat_update)
            except:
                self.current_rot_test = quat_update
            self.vispg(self.current_rot_test, visualize=True, name="integrated rot quat")  # seems correct

        if self.visualize == 3:
            try:
                self.current_rot_test = self.current_rot_test.dot(pos.rotation_matrix)
            except:
                self.current_rot_test = pos.rotation_matrix
            self.vispg(rotMat_to_quaternion(self.current_rot_test), visualize=True, name="integrated rot") # correct with quaternion_to_rotMat

        # Extraction of the angle axis from the rotation matrix
        [pos.roll, pos.pitch, pos.yaw] = rotMat_to_anlgeAxis(pos.rotation_matrix)

        # Save and update the rotation
        current_rot = rot_first
        # Displacement
        for i in range(1, len(frames)):
            dt = (frames[i].ts - frames[i-1].ts) / 1000
            dt2 = dt * dt

            current_rot = current_rot.dot(quaternion_to_rotMat(frames[0].quaternion))

            # get the acceleration in the primary (starting) frame
            [ax_, ay_, az_] = current_rot.dot([frames[i].ax, frames[i].ay, frames[i].az]) # TO check

            pos.x += self.velocity.x * dt + 0.5 * ax_ * dt2
            pos.y += self.velocity.y * dt + 0.5 * ay_ * dt2
            pos.z += self.velocity.z * dt + 0.5 * az_ * dt2

            self.velocity.x += ax_ * dt
            self.velocity.y += ay_ * dt
            self.velocity.z += az_ * dt

            # print("dt:", dt)

            self.velocity.dampen()

            self.logger.debug(f"pos calculated {pos}")

            # print("intergrate : current rot", current_rot)
            # print(f"accel in: {[frames[i].ax, frames[i].ay, frames[i].az]}, accel rotated: {[ax_, ay_, az_]}")
            # print(f"velocity: {self.velocity.x, self.velocity.y, self.velocity.z}")
        return pos

    def find_integration_frames(self, ts0, ts1, i0, i1) -> List[IMUFrame]:
        integration_frames: List[IMUFrame] = [] # IMU data with quaternion for current rotation
        lower_frame_bound: IMUFrame = interpolate_frames(self.imu_buffer[i0], self.imu_buffer[i0 + 1], ts0)
        integration_frames.append(lower_frame_bound)

        for j in range(i0+1, i1-1):
            integration_frames.append(self.imu_buffer[j])

        upper_frame_bound: IMUFrame = interpolate_frames(self.imu_buffer[i1 - 1], self.imu_buffer[i1], ts1)
        if self.visualize == 2:
            self.vispg(upper_frame_bound.quaternion, visualize=True, name="upper_frame_bound")
        integration_frames.append(upper_frame_bound)
        return integration_frames

    def choose_nearest_homography(self, vo_result: VOResult, imu_homog: Homography) -> np.array:
        imu_homog_matrix = imu_homog.as_Tmatrix()
        imu_rot = imu_homog_matrix[0:3, 0:3]
        imu_t_vec = imu_homog_matrix[0:3, 3]

        best_match = (None, None, np.inf, None)

        for homog in vo_result.homogs:
            k_t = 0.0

            # Correction: Homography gives a result rotated from our camera coordinate frame.
            vo_to_camera = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]])
            # Expressing the translation vector in the camera frame
            vo_t_vec = vo_to_camera.dot(homog[0:3, 3])
            # Expressing the rotation in the camera frame
            # normalise the input frame as it happens that the rotation matrix has elements with value above 1
            vo_homog = normalise_rotation(homog[0:3, 0:3])
            vo_angle_axis = vo_to_camera.dot(rotMat_to_anlgeAxis(vo_homog))
            vo_quat = angleAxis_to_quaternion(vo_angle_axis)

            # The rotation expressed as a rotation matrix lacks in performance
            vo_rot = angleAxis_to_rotMat(vo_angle_axis)
            vo_rot = normalise_rotation(vo_rot)

            # if vo_rot.shape != (3, 3):
            #     print('vo_to_camera', vo_to_camera)
            #     print('homog[0:3, 0:3]\n', homog[0:3, 0:3])
            #     print('vo_homog', vo_homog)
            #     print('vo_t_vec', vo_t_vec)
            #     print('vo_angle_axis', vo_angle_axis)
            #     print('vo_rot.shape', vo_rot.shape)

            # print("000vo_rot original\n", homog[0:3, 0:3])
            # print("angle axis", rotMat_to_anlgeAxis(homog[0:3, 0:3]))
            # print("rotated angle axis", vo_to_camera.dot(rotMat_to_anlgeAxis(homog[0:3, 0:3])))
            # print("vo_rot _cam\n", vo_rot, "\n angle axis new rot", rotMat_to_anlgeAxis(vo_rot))
            # print("ypr", rotMat_to_ypr(vo_rot))

            ## second way of calculating it
            # vo_rot = angleAxis_to_rotMat(quaternion_apply(rotMat_to_quaternion(vo_to_camera), rotMat_to_anlgeAxis(homog[0:3, 0:3])))
            # vo_rot = normalise_rotation(vo_rot)
            #
            # print("111vo_rot original\n", homog[0:3, 0:3])
            # print("angle axis", rotMat_to_anlgeAxis(homog[0:3, 0:3]))
            # print("rotated angle axis", quaternion_apply(rotMat_to_quaternion(vo_to_camera), rotMat_to_anlgeAxis(homog[0:3, 0:3])))
            # print("vo_rot _cam\n", vo_rot, "\n angle axis new rot", rotMat_to_anlgeAxis(vo_rot))
            # print("\n ypr", rotMat_to_ypr(vo_rot))

            # Exit if the rotation matrix is not computed as expected
            check_correct_rot_mat(vo_rot)

            # Robot dynamic script p51
            # Issues for rotation angles close to zero or pi
            delta_rot = rotMat_to_anlgeAxis(imu_rot.T.dot(vo_rot))
            distance = np.linalg.norm(delta_rot) + k_t * np.linalg.norm(imu_t_vec - vo_t_vec)

            if distance < best_match[2]:
                best_match = (vo_quat, vo_t_vec, distance, delta_rot)

        imu_ypr = rotMat_to_ypr(imu_homog_matrix[..., :3])
        vo_ypr = quat_to_ypr(best_match[0])
        imu_xyz = str(imu_homog_matrix[..., 3:]).replace("\n", "")
        vo_xyz = str(best_match[1]).replace("\n", "")
        self.logger.debug(f"Prediction Offset:\n"
                         f"IMU Euler [yaw, pitch, roll] angles :\n{imu_ypr}\nVO angles :\n{vo_ypr}, (RAD)\n"
                         f"IMU pos :{imu_xyz}\nVO pos  :{vo_xyz}")

        if self.visualize == 10:
            self.vispg(best_match[0], visualize=True, name="best match found")
            pass

        #PLOT
        # visualize_distance_metric(best_match, degrees, imu_angles, vo_angles)
        self.logger.debug(f'returning {np.column_stack((quaternion_to_rotMat(best_match[0]), best_match[1]))}')
        # TODO would be better working with quaternion
        # return  best_match[0], best_match[1] # returning the quaternion and translation
        return np.column_stack((quaternion_to_rotMat(best_match[0]), best_match[1]))


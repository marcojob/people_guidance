import io
import platform
import re
from math import tan, atan2, cos, sin, pi, sqrt
import logging
from time import sleep, monotonic
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import List, Optional, Dict
from queue import Queue
from pathlib import Path
import copy
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .utils import *
from ..drivers_module import ACCEL_G
from ..module import Module
from ...utils import DEFAULT_DATASET
from .position import Position, new_empty_position, new_interpolated_position

# TODO: IMU data (acceleration) needs to be multiplied by (-1) to compensate for calculation errors in previous datasets
# TODO: Remove the hardcoded change in this file for further datasets recorded after the correction.
HARDCODED_NEGATIVE_CORRECTION = 1 # |-1| for datasets_3 with old IMU data or earlier datasets

''' 
Coordinates valid for the IMU only : looking at the front of the camera,  x points upwards, 
y horizontally to the right, z horizontally towards the camera, towards the plane.
Output and calculations in camera coordinate system: looking at the front of the camera,  z points upwards, 
y horizontally to the right, x horizontally from the camera, out of the plane.
In Camera coordinates: Z = X_IMU, X = -Z_IMU, Y = Y_IMU (-90° rotation around the Y axis)
'''


# TODO: gravity compensation occurs in the starting frame. Averaging the first few elements for defining precisely the
#  direction of the gravity and setting the reference coordinate frame can help

class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module",
                                                       outputs=[("position_vis", 10), ("position", 10)],
                                                       inputs=["drivers_module:accelerations"],
                                                       services=["position_request"],
                                                       log_dir=log_dir)
        self.args = args

        self.drift_tracking: Dict = self.reset_drift_tracker()
        self.pos: Position = new_empty_position()
        self.tracked_positions: List[Position] = []
        self.prev_imu_frame: Optional[IMUFrame] = None
        self.last_visualized_pos: Optional[Position] = None

        self.event_timestamps: Dict[str, float] = {}  # keep track of when events (classified by name) happened last
        self.speed: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.acceleration: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0, "x_last": 0.0, "y_last": 0.0, "z_last": 0.0}

        self.counter_same_request = 0

    def start(self):
        # TODO: evaluate position quality
        self.services["position_request"].register_handler(self.position_request)

        while True:
            self.handle_data()
            self.handle_requests()

    def handle_data(self):
        input_data = self.get("drivers_module:accelerations")
        if not input_data:
            sleep(0.0001)
        else:
            frame = self.frame_from_input_data(input_data)  # m/s^2 // °/s to rad/s

            if self.prev_imu_frame is None:
                # if the frame we just received is the first one we have received.
                self.prev_imu_frame = frame  # TODO: check where used. should use rotated elements instead?
            else:
                # TODO: do not track and do not update if the timestamp did not change (dt = 0)
                # ERROR description : 91426359 appears twice in the printed list (old dataset_3).
                self.update_position(frame)
                self.append_tracked_positions()

                # Display in a scatter plot (debug)
                curr_time = monotonic()
                msg_name = "last_visualization"
                if not MEASURE_SUMMED_ERROR_ACC_AUTO or (self.drift_tracking["auto_state"] and MEASURE_SUMMED_ERROR_ACC_AUTO):
                    if VISUALIZE_LOCALLY and (msg_name not in self.event_timestamps or \
                            (curr_time - self.event_timestamps[msg_name]) * VISUALIZE_LOCALLY_FREQ > 1 or \
                            VISUALIZE_LOCALLY_FREQ > 99):
                        self.logger.info(f"Visualization at current time {curr_time} for frame {frame}")
                        self.event_timestamps[msg_name] = curr_time
                        visualize_locally(self.pos, frame, self.drift_tracking, self.acceleration, plot_pos=True,
                                          plot_angles=True, plot_acc_input=True, plot_acc_transformed=True)

            self.publish_to_visualization()

    @staticmethod
    def frame_from_input_data(input_data: Dict) -> IMUFrame:
        # In Camera coordinates: Z = X_IMU, X = -Z_IMU, Y = Y_IMU (-90° rotation around the Y axis)
        return IMUFrame(
            ax=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_z']),
            # ^Correction, no sign '-' for dataset 3 & onwards needed
            ay=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_y']),
            az=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_x']),
            gx=-float(input_data['data']['gyro_z']) * pi / 180,
            gy=float(input_data['data']['gyro_y']) * pi / 180,
            gz=float(input_data['data']['gyro_x']) * pi / 180,
            ts=input_data['timestamp']
        )

    def append_tracked_positions(self):
        self.tracked_positions.append(copy.deepcopy(self.pos))

        if len(self.tracked_positions) > 300:
            self.tracked_positions.pop(0)

    def update_position(self, frame: IMUFrame) -> None:
        dt: float = (frame.ts - self.prev_imu_frame.ts) / 1000.0
        self.pos.ts = frame.ts
        self.complementary_filter_new(frame, dt)
        self.position_estimation_simple(frame, dt)
        self.prev_imu_frame = frame

    @staticmethod
    def reset_drift_tracker():
        return {
            "total_time": 0.0,
            "n_elt_summed": 0,
            "total_acc_x": 0,
            "total_acc_y": 0,
            "total_acc_z": 0,
            "total_acc_avg_x": 0,
            "total_acc_avg_y": 0,
            "total_acc_avg_z": 0,
            "total_g_x": 0,
            "total_g_y": 0,
            "total_g_z": 0,
            "total_g_avg_x": 0,
            "total_g_avg_y": 0,
            "total_g_avg_z": 0,
            "auto_state": 0,   # start at 0, changes to 1 when done measuring
            "auto_result": [0, 0, 0] # correction result of automatic measurements
        }

    def complementary_filter(self, frame: IMUFrame, dt: float):  # TODO: check formulas
        # Only integrate for the yaw
        self.pos.yaw += frame.gz * dt # |not ok|
        # Compensate for drift with accelerometer data https: // philsal.co.uk / projects / imu - attitude - estimation
        roll_accel = atan2(frame.ay, sqrt(frame.ax ** 2 + frame.az ** 2)) # scalar to rad |ok|
        pitch_accel = atan2(-frame.ax, sqrt(frame.ay ** 2 + frame.az ** 2)) # scalar to rad |needs a -|
        # Gyro data
        roll_vel_gyro = frame.gz + (frame.gz * sin(self.pos.roll) + frame.gz * cos(self.pos.roll)) * tan(self.pos.pitch)
        pitch_vel_gyro = frame.gy * cos(self.pos.roll) - frame.gz * sin(self.pos.roll)
        # Update estimation

        self.pos.roll = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * (self.pos.roll + roll_vel_gyro * dt) \
                        + ALPHA_COMPLEMENTARY_FILTER * roll_accel
        self.pos.pitch = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * self.pos.pitch + pitch_vel_gyro * dt \
                         + ALPHA_COMPLEMENTARY_FILTER * pitch_accel

    def complementary_filter_new(self, frame: IMUFrame, dt: float):  # TODO: check formulas
        roll_accel_hat = atan2(frame.ay, sqrt(frame.ax ** 2 + frame.az ** 2))  # scalar to rad |ok|
        pitch_accel_hat = atan2(-frame.ax, sqrt(frame.ay ** 2 + frame.az ** 2))  # scalar to rad |ok|

        #TODO : g drehen???

        roll_gyr_hat = self.pos.roll + dt * (frame.gx + sin(self.pos.roll) * tan(self.pos.pitch) * frame.gy
                                             + cos(self.pos.roll) * tan(self.pos.pitch) * frame.gz) # not sure
        pitch_gyr_hat = self.pos.pitch + dt * (cos(self.pos.roll) * frame.gy - sin(self.pos.roll) * frame.gz) # not sure

        math_cos_pitch = cos(self.pos.pitch)
        if not round(math_cos_pitch, 4) : # no division by 0 allowed
            math_cos_pitch = 0.001
        self.pos.yaw = self.pos.yaw + dt * (sin(self.pos.roll) / math_cos_pitch * frame.gy
                                            + cos(self.pos.roll) / math_cos_pitch * frame.gz) # not sure

        self.pos.roll = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * roll_gyr_hat + ALPHA_COMPLEMENTARY_FILTER * roll_accel_hat # |ok|
        self.pos.pitch = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * pitch_gyr_hat + ALPHA_COMPLEMENTARY_FILTER * pitch_accel_hat # |ok|

    def position_estimation_simple(self, frame: IMUFrame, dt: float):
        # Integrate the acceleration after transforming the acceleration in world coordinates
        if METHOD_ERROR_ACC_CORRECTION:
            corr_acc = CORRECTION_ACC
        if METHOD_ERROR_ACC_CORRECTION_AUTO:
            corr_acc = self.drift_tracking["auto_result"]

        [self.acceleration["x_alt"], self.acceleration["x_alt"], self.acceleration["x_alt"]] = [
            self.acceleration["x"], self.acceleration["y"], self.acceleration["z"]
        ]

        if METHOD_SCIPY_ROTATION:
            self.acceleration_after_rotation_gravity_compensation_correction(frame, corr_from_measurements=CORRECTION_ACC)
        else:
            acc_vector_imu_frame = self.rotation_coordinate_frame([frame.ax, frame.ay, frame.az], way="IMU_to_inertial")
            self.gravity_compensation_correction(acc_vector_imu_frame, corr_from_measurements=CORRECTION_ACC)

        #self.direct_2_integration()

        # Measure the sum of the deviation from 0
        self.drift_measurement(dt)

        # Reduce the velocity to reduce the drift
        self.dampen_velocity()

        # Integrate to get the position
        self.pos.x += self.speed["x"] * dt + 0.5 * self.acceleration["x"] * dt * dt
        self.pos.y += self.speed["y"] * dt + 0.5 * self.acceleration["y"] * dt * dt
        self.pos.z += self.speed["z"] * dt + 0.5 * self.acceleration["z"] * dt * dt

        # Update speed
        self.speed["x"] += self.acceleration["x"] * dt
        self.speed["y"] += self.acceleration["y"] * dt
        self.speed["z"] += self.acceleration["z"] * dt

        self.const_frequency_log("new_element_and_timestamp", POSITION_PUBLISH_NEW_TIMESTAMP,
                                 f"Position updated at timestamp: {frame.ts}, time interval: {dt}",
                                 )
        self.const_frequency_log("last_summed_acc", POSITION_PUBLISH_ACC_FREQ,
                                 f"Acceleration after rotation :  {[self.acceleration['x'], self.acceleration['y'], self.acceleration['z']]}, "
                                 f"Corrected speed : {[self.speed['x'], self.speed['y'], self.speed['z']]} "
                                 )

    def direct_2_integration(self):
        if NO_VELOCITY_ONLY_ACC_INTEGRATION:
            self.speed["x"] = 0
            self.speed["y"] = 0
            self.speed["z"] = 0

    def rotation_coordinate_frame(self, vector: List, way="none") -> List:
        # rotate a given vector expressed in the inertial coordinate system so that it is expressed in the IMU coordinate system
        # rotation around the axis of the inertial frame!
        # Radians only!
        # Rotations possible: way="inertial_to_IMU", way="IMU_to_inertial"
        if len(vector) != 3:
            print("Wrong vector input for rotation, returning original input vector")
            return vector
        [xx, yy, zz] = [self.pos.roll, self.pos.pitch, self.pos.yaw]
        x = np.array(
            [[1,    0.,         0.],
             [0.,   cos(xx),    -sin(xx)],
             [0,    sin(xx),    cos(xx)]])
        y = np.array(
            [[cos(yy),  0., sin(yy)],
             [0.,       1,  0],
             [-sin(yy), 0,  cos(yy)]])
        z = np.array(
            [[cos(zz),  -sin(zz),   0.],
             [sin(zz),  cos(zz),    0],
             [0,        0,          1]])
        rotation = np.round(np.matmul(np.matmul(x, y), z), 6)
        if way=="inertial_to_IMU":
            # forward, all good
            pass
        elif way=="IMU_to_inertial":
            # backward, need to transpose
            rotation = rotation.transpose()
        else:
            self.logger.critical("no compatible rotation direction given, returning forward rotation")
        return rotation.dot(vector)

    def gravity_compensation_correction(self, vector, corr_from_measurements) -> None:
        # rotation to starting coordinates and gravity compensation
        [self.acceleration["x"], self.acceleration["y"], self.acceleration["z"]] = vector - [0, 0, ACCEL_G] - corr_from_measurements #TODO: corr gyro

    def acceleration_after_rotation_gravity_compensation_correction(self, frame: IMUFrame, corr_from_measurements):
        r = R.from_euler('xyz', [self.pos.roll, self.pos.pitch, self.pos.yaw], degrees=False).inv() # Rad!
        # rotation to starting coordinates and gravity compensation
        [self.acceleration["x"], self.acceleration["y"], self.acceleration["z"]] = r.apply([frame.ax, frame.ay, frame.az]) \
                                                                                   - [0, 0, ACCEL_G] - corr_from_measurements

    def drift_measurement(self, dt: float) -> None:  # TODO add gyro reset
        # Calculate the mean for drift compensation
        if MEASURE_SUMMED_ERROR_ACC:
            self.drift_tracking["total_time"] += dt
            self.drift_tracking["n_elt_summed"] += 1
            self.drift_tracking["total_acc_x"] += self.acceleration["x"]
            self.drift_tracking["total_acc_y"] += self.acceleration["y"]
            self.drift_tracking["total_acc_z"] += self.acceleration["z"]
            self.drift_tracking["total_acc_avg_x"] = round(
                self.drift_tracking["total_acc_x"] / self.drift_tracking["n_elt_summed"], 3)
            self.drift_tracking["total_acc_avg_y"] = round(
                self.drift_tracking["total_acc_y"] / self.drift_tracking["n_elt_summed"], 3)
            self.drift_tracking["total_acc_avg_z"] = round(
                self.drift_tracking["total_acc_z"] / self.drift_tracking["n_elt_summed"], 3)

            self.const_frequency_log("log_drift_stats", PUBLISH_SUMMED_MEASURE_ERROR_ACC,
                                     f"Drift {self.drift_tracking}")

        elif MEASURE_SUMMED_ERROR_ACC_AUTO and not self.drift_tracking["auto_state"]:
            self.drift_tracking["total_time"] += dt
            if self.drift_tracking["total_time"] > MEASURE_ERROR_TIME_START:
                if self.drift_tracking["total_time"] < MEASURE_ERROR_TIME_STOP:
                    self.drift_tracking["n_elt_summed"] += 1
                    self.drift_tracking["total_acc_x"] += self.acceleration["x"]
                    self.drift_tracking["total_acc_y"] += self.acceleration["y"]
                    self.drift_tracking["total_acc_z"] += self.acceleration["z"]
                    self.drift_tracking["total_acc_avg_x"] = round(
                        self.drift_tracking["total_acc_x"] / self.drift_tracking["n_elt_summed"], 6)
                    self.drift_tracking["total_acc_avg_y"] = round(
                        self.drift_tracking["total_acc_y"] / self.drift_tracking["n_elt_summed"], 6)
                    self.drift_tracking["total_acc_avg_z"] = round(
                        self.drift_tracking["total_acc_z"] / self.drift_tracking["n_elt_summed"], 6)
                    # self.drift_tracking["total_g_x"] += self.acceleration["x"]
                    # self.drift_tracking["total_g_y"] += self.acceleration["y"]
                    # self.drift_tracking["total_g_z"] += self.acceleration["z"]
                    # self.drift_tracking["total_g_avg_x"] = round(
                    #     self.drift_tracking["total_g_x"] / self.drift_tracking["n_elt_summed"], 6)
                    # self.drift_tracking["total_g_avg_y"] = round(
                    #     self.drift_tracking["total_g_y"] / self.drift_tracking["n_elt_summed"], 6)
                    # self.drift_tracking["total_g_avg_z"] = round(
                    #     self.drift_tracking["total_g_z"] / self.drift_tracking["n_elt_summed"], 6)
                else:
                    # done measuring
                    self.drift_tracking["auto_state"] = 1
                    self.drift_tracking["auto_result"] = [self.drift_tracking["total_acc_avg_x"],
                                                          self.drift_tracking["total_acc_avg_y"],
                                                          self.drift_tracking["total_acc_avg_z"]]
                    self.logger.info(f"Automatic measurement done, correction : {self.drift_tracking['auto_result']}")
                    self.speed["x"], self.speed["y"], self.speed["z"] = 0, 0, 0
                    self.pos.x, self.pos.y, self.pos.z = 0, 0, 0
                    self.pos.roll = 0.0
                    self.pos.pitch = 0.0
                    self.pos.yaw = 0.0
                    self.logger.info(f"Done_measuring Drift - AUTOMATIC {self.drift_tracking}")

                self.const_frequency_log("log_drift_stats", PUBLISH_SUMMED_MEASURE_ERROR_ACC,
                                         f"Drift - AUTOMATIC {self.drift_tracking}")

    def dampen_velocity(self) -> None:  # TODO : reset dependent on dt
        if METHOD_RESET_VELOCITY:
            curr_time = monotonic()
            if "velocity_reset" not in self.event_timestamps or \
                    (curr_time - self.event_timestamps["velocity_reset"]) * RESET_VEL_FREQ > 1 or \
                    RESET_VEL_FREQ > 99:
                self.speed["x"] *= RESET_VEL_FREQ_COEF_X
                self.speed["y"] *= RESET_VEL_FREQ_COEF_Y
                self.speed["z"] *= RESET_VEL_FREQ_COEF_Z

                self.event_timestamps["velocity_reset"] = curr_time

    def limit_velocity(self) -> None:
        # Walking velocity cannot be above 2m/s # TODO
        pass

    def publish_to_visualization(self) -> None:
        curr_time = monotonic()
        if "last_visu_pub" not in self.event_timestamps or self.last_visualized_pos is None \
                or (curr_time - self.event_timestamps["last_visu_pub"]) * POSITION_PUBLISH_FREQ > 1\
                or POSITION_PUBLISH_FREQ > 99:

            if self.pos != self.last_visualized_pos:
                self.publish("position", self.pos.__dict__, POS_VALIDITY_MS)
                self.last_visualized_pos = copy.deepcopy(self.pos)
                self.event_timestamps["last_visu_pub"] = curr_time

    def const_frequency_log(self, msg_name: str, frequency: int, msg: str, level=logging.INFO) -> None:
        curr_time = monotonic()
        if msg_name not in self.event_timestamps or \
                (curr_time - self.event_timestamps[msg_name]) * frequency > 1 or \
                frequency > 99:
            self.logger.info(msg)
            self.event_timestamps[msg_name] = curr_time

    def position_request(self, request): # TODO: check timeout errors

        requested_timestamp = request["payload"]
        neighbors = [None, None]

        self.logger.debug(f"{len(self.tracked_positions)}  - {[str(pos.ts) for pos in self.tracked_positions]}")

        for idx, position in enumerate(self.tracked_positions):
            # this assumes that our positions are sorted old to new.
            if position.ts <= requested_timestamp:
                neighbors[0] = (idx, position)
            if position.ts >= requested_timestamp:
                neighbors[1] = (idx, position)
                break

        if neighbors[0] is not None and neighbors[1] is not None:
            self.counter_same_request = 0
            self.logger.debug(f"Normal interpolation, asking for ts {requested_timestamp}")
            interp_position = new_interpolated_position(requested_timestamp, neighbors[0][1], neighbors[1][1])
            return interp_position.__dict__

        elif neighbors[0] is None and neighbors[1] is not None and neighbors[1][0] < len(self.tracked_positions) - 1:
            self.counter_same_request = 0
            self.logger.critical("Extrapolating backwards!")
            interp_position = new_interpolated_position(requested_timestamp, neighbors[1][1],
                                                        self.tracked_positions[neighbors[1][0] + 1])
            return interp_position.__dict__

        else:
            offset = self.pos.ts - requested_timestamp
            self.logger.info(f"Could not interpolate for position with timestamp {requested_timestamp}. "
                             f"Current offset {offset}, asked {self.counter_same_request} times")
            if self.counter_same_request > 200: #TODO: REMOVE Quick fix
                self.logger.warning(f"Not interpolating. Quick fix. requested {requested_timestamp}, "
                                    f"sending {self.tracked_positions[-1]}")
                self.counter_same_request = 0
                return self.tracked_positions[-1]
            self.counter_same_request += 1
            return None

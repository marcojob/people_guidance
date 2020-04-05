import io
import platform
import re
import math
from time import sleep, monotonic, perf_counter
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import List
from queue import Queue
from pathlib import Path
import copy

from .utils import *
from ..drivers_module import ACCEL_G
from ..module import Module
from ...utils import DEFAULT_DATASET
from .position import Position

# Position estimation based on the data from the accelerometer and the gyroscope
class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[("position", 10)],
                                                       inputs=["drivers_module:accelerations"],
                                                       services=["position_request"],
                                                       log_dir=log_dir)
        self.args = args

        # General inits
        self.count_valid_input = 0  # Number of inputs processed
        self.countall = 0  # Number of loops since start
        self.count_outputs = 0  # Number of elements published
        # Timestamps
        self.timestamp_last_input = 0  # time last input received
        self.timestamp_last_output = monotonic()  # time last output published
        self.timestamp_last_displayed_input = monotonic()
        self.timestamp_last_displayed_acc = monotonic()
        self.timestamp_last_reset_vel = monotonic()
        self.timestamp_last_summed_acc = monotonic()
        self.loop_time = monotonic()  # time start of loop

        # Initialization and tracking
        self.dt_initialised = False
        self.last_data_dict_published = None
        # Tracking drift
        self.total_time = 0
        self.total_number_elt_summed = 0
        self.total_acc_x = 0
        self.total_acc_y = 0
        self.total_acc_z = 0


        # Output_speed (m/s)
        self.speed_x = 0.
        self.speed_y = 0.
        self.speed_z = 0.
        # Timestamp element

        self.pos = Position.new_empty()
        self.tracked_positions: List[Position] = []

    def start(self):
        if DEBUG_POSITION >= 1:
            self.logger.info("Starting position_estimation_module...")

        self.services["position_request"].register_handler(self.position_request)

        while True:
            # Retrieve data
            input_data = self.get("drivers_module:accelerations")

            # Handle requests
            self.handle_requests()

            if DEBUG_POSITION > 1:
                self.countall += 1  # count number of time the loop gets executed
                if DEBUG_POSITION >= 3:
                    self.loop_time = monotonic()
                    if DEBUG_POSITION == 4:
                        self.logger.info("loop time : {:.4f}".format(self.loop_time))

            if input_data:  # m/s^2 // °/s
                accel_x = float(input_data['data']['accel_x'])
                accel_y = float(input_data['data']['accel_y'])
                accel_z = float(input_data['data']['accel_z'])
                gyro_x = float(input_data['data']['gyro_x']) * math.pi / 180
                gyro_y = float(input_data['data']['gyro_y']) * math.pi / 180
                gyro_z = float(input_data['data']['gyro_z']) * math.pi / 180
                timestamp = input_data['timestamp']
                validity = input_data['validity']

                self.debug_input_data(input_data, timestamp)
                if (monotonic() - self.timestamp_last_displayed_input) * POSITION_PUBLISH_INPUT_FREQ > 1:
                    self.logger.info("Data received :  {}".format(input_data))
                    self.timestamp_last_displayed_input = monotonic()

                # Delta time since last input
                dt = self.input_data_dt(timestamp)

                # Filter input, remove gravity, compute position and save the data
                self.complementary_filter(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt)
                self.position_estimation_simple(accel_x, accel_y, accel_z, dt)

                self.update_tracked_positions()

            # TODO: evaluate position quality
            # TODO: save last few estimations with absolute timestamp

            # Downsample to POSITION_PUBLISH_FREQ (Hz) and publish
            if (monotonic() - self.timestamp_last_output) * POSITION_PUBLISH_FREQ > 1:
                self.downsample_publish()

            # Time for processing
            if DEBUG_POSITION > 3:  # Full debug
                self.logger.info("Time needed for the loop : {:.4f} s. "
                                 .format((monotonic() - self.loop_time)))

    def update_tracked_positions(self):
        self.tracked_positions.append(copy.deepcopy(self.pos))

        if len(self.tracked_positions) > 300:
            self.tracked_positions.pop(0)

    # FUNCTION IMPLEMENTATION
    # dt time calculation between two consecutive analyzed samples. Input : ms, output : s
    def input_data_dt(self, timestamp):
        dt = (timestamp - self.pos.timestamp) / 1000

        if DEBUG_POSITION > 2:
            self.logger.info("Time between elements : dt = {}. ".format(dt))
            self.logger.info("Timestamp, {}, self.pos.timestamp : {}. ".format(timestamp, self.pos.timestamp))

        if self.pos.timestamp != 0 and dt < 1:  # dt has not a too high value
            # Update
            self.pos.timestamp = timestamp
            return dt  # seconds
        elif DEBUG_POSITION > 1:
            self.logger.info("dt error: dt = {}.".format(dt))

        if self.dt_initialised == True:
            self.logger.warning("Corrupted calculation delta time, returning 10ms, saving timestamp")
        else:
            self.dt_initialised = True
            self.logger.info("delta time error, returning 10ms, saving timestamp")
        # Update
        self.pos.timestamp = timestamp
        return 0.01

    def complementary_filter(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt):
        # Only integrate for the yaw
        self.pos.yaw += gyro_z * dt
        # Compensate for drift with accelerometer data
        roll_accel = math.atan(accel_y / math.sqrt(accel_x ** 2 + accel_z ** 2))
        pitch_accel = math.atan(accel_x / math.sqrt(accel_y ** 2 + accel_z ** 2))
        # Gyro data
        a = gyro_y * math.sin(self.pos.roll) + gyro_z * math.cos(self.pos.roll)
        roll_vel_gyro = gyro_x + a * math.tan(self.pos.pitch)
        pitch_vel_gyro = gyro_y * math.cos(self.pos.roll) - gyro_z * math.sin(self.pos.roll)
        # Update estimation
        b = self.pos.roll + roll_vel_gyro * dt
        self.pos.roll = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * b + ALPHA_COMPLEMENTARY_FILTER * roll_accel
        c = self.pos.pitch + pitch_vel_gyro * dt
        self.pos.pitch = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * c + ALPHA_COMPLEMENTARY_FILTER * pitch_accel

    def position_estimation_simple(self,
                                   accel_x: float, accel_y: float, accel_z: float,
                                   dt: float):
        if dt > 0:
            # Integrate the acceleration after transforming the acceleration in world coordinates
            r = R.from_euler('xyz', [self.pos.roll, self.pos.pitch, self.pos.yaw], degrees=True)
            accel_rot = r.apply([accel_x, accel_y, accel_z])
            self.speed_x += (accel_rot[0] - CORRECTION_ACC[0]) * dt
            self.speed_y += (accel_rot[1] - CORRECTION_ACC[1]) * dt
            self.speed_z += (accel_rot[2] + ACCEL_G - CORRECTION_ACC[2]) * dt  # TODO : check coordinate system setup
            # Calculate the mean for drift compensation
            if MEASURE_SUMMED_ERROR_ACC:
                self.total_time += dt
                self.total_number_elt_summed += 1
                self.total_acc_x += accel_rot[0]
                self.total_acc_y += accel_rot[1]
                self.total_acc_z += accel_rot[2] + ACCEL_G
                if (monotonic() - self.timestamp_last_summed_acc) * PUBLISH_SUMMED_MEASURE_ERROR_ACC > 1:
                    self.logger.info("Sum dt : {}, Number of elements : {}, Sum Acc : {} "
                                     .format(self.total_time, self.total_number_elt_summed,
                                             [self.total_acc_x, self.total_acc_y, self.total_acc_z]))
                    self.timestamp_last_summed_acc = monotonic()
            # Reduce the velocity to reduce the drift
            if METHOD_RESET_VELOCITY and (monotonic() - self.timestamp_last_reset_vel) * RESET_VEL_FREQ > 1:
                self.speed_x *= RESET_VEL_FREQ_COEF_X
                self.speed_y *= RESET_VEL_FREQ_COEF_Y
                self.speed_z *= RESET_VEL_FREQ_COEF_Z
                self.timestamp_last_reset_vel = monotonic()
            # Integrate to get the position
            self.pos.x += self.speed_x * dt
            self.pos.y += self.speed_y * dt
            self.pos.z += self.speed_z * dt
            if (monotonic() - self.timestamp_last_displayed_acc) * POSITION_PUBLISH_ACC_FREQ > 1:
                self.logger.info("Acceleration after rotation :  {}, Corrected speed : {} "
                                 .format(accel_rot, [self.speed_x, self.speed_y, self.speed_z]))
                self.timestamp_last_displayed_acc = monotonic()

    def downsample_publish(self):
        data_dict = {'pos_x': self.pos.x,
                     'pos_y': self.pos.y,
                     'pos_z': self.pos.z,
                     'angle_x': self.pos.roll,
                     'angle_y': self.pos.pitch,
                     'angle_z': self.pos.yaw
                     }
        if data_dict != self.last_data_dict_published:
            self.debug_downsample_publish(data_dict)

            # Publish with the timestamp of the last element received and update timestamp
            self.publish("position", data_dict, POS_VALIDITY_MS)
            self.last_data_dict_published = data_dict
        # Update
        self.timestamp_last_output = monotonic()

    # TODO: answer to position requests interpolating between saved data

    # DEBUG FUNCTIONS
    def debug_input_data(self, input_data, timestamp):
        if DEBUG_POSITION > 1:
            # Counter of valid values
            self.count_valid_input += 1

            # Difference in time from the timestamp of the data
            if self.timestamp_last_input == 0:
                timeDelta = 0
            else:
                timeDelta = timestamp - self.timestamp_last_input

            # Log output for each element received
            self.logger.info("Received Acceleration element N° {} within {} loops, time between samples : {}. "
                             .format(self.count_valid_input, self.countall, timeDelta))
            self.timestamp_last_input = timestamp

            if DEBUG_POSITION >= 3:
                self.logger.info("Data :  {}".format(input_data))

    def debug_downsample_publish(self, data_dict):
        if DEBUG_POSITION > 1:
            # Counter of valid values
            self.count_outputs += 1

            # Log output for each element sent
            self.logger.info("Sent data N° {}, time between samples : {:.4f} seconds. "
                             .format(self.count_outputs, monotonic() - self.timestamp_last_output))

            if DEBUG_POSITION >= 3:
                self.logger.info("Data sent :  {}".format(data_dict))
        # self.logger.info("Data sent :  {}".format(data_dict))

    def position_request(self, request):

        requested_timestamp = request["payload"]

        neighbors = [None, None]

        for position in reversed(self.tracked_positions):
            # start with the newest position
            if position.timestamp <= requested_timestamp:
                neighbors[0] = position
                break

        for position in self.tracked_positions:
            # start with the oldest position
            if position.timestamp >= requested_timestamp:
                neighbors[1] = position
                break

        if neighbors[0] is not None and neighbors[1] is not None:
            interp_position = Position.new_interpolate(requested_timestamp, neighbors[0], neighbors[1])
            return {"id": request["id"], "payload": interp_position.__dict__}
        else:
            self.logger.info(f"Could not interpolate for position with timestamp {requested_timestamp}."
                             f"Current timestamp {self.pos.timestamp}, {neighbors}")
            return None

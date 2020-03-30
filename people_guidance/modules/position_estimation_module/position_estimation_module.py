import io
import platform
import re
from math import atan2
from queue import Queue
from time import sleep, monotonic, perf_counter
from pathlib import Path

from .utils import *
from ..drivers_module import ACCEL_RANGE
from ..module import Module
from ...utils import DEFAULT_DATASET

# Read data IMU
# Compute Position and save to array
# Answer to position requests (interpolating)

class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[("position", 10)],
                                            inputs=["drivers_module:accelerations"], log_dir=log_dir)
        self.args = args

        # General inits
        self.count_valid_input = 0  # Number of inputs processed
        self.countall = 0           # Number of loops since start
        self.count_outputs = 0      # Number of elements published
        # Timestamps
        self.timestamp_last_input = 0                       # time last input received
        self.timestamp_last_output = self.get_time_ns()     # time last output published
        self.loop_time = self.get_time_ns()                 # time start of loop

        self.roll = 0.
        self.pitch = 0.

        # Output
        self.pos_x = 0.
        self.pos_y = 0.
        self.pos_z = 0.
        # Output_speed
        self.speed_x = 0.
        self.speed_y = 0.
        self.speed_z = 0.
        # Timestamp element
        self.timestamp:int = 0
        # Roll Pitch Yaw
        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0.

    def start(self):
        if DEBUG_POSITION >= 1:
            self.logger.info("Starting position_estimation_module...")

        while(True):
            # Retrieve data
            input_data = self.get("drivers_module:accelerations")

            if DEBUG_POSITION > 1:
                self.countall += 1  # count number of time the loop gets executed
                if DEBUG_POSITION == 3:  # Full debug
                    self.loop_time = self.get_time_ns()

            if input_data: # m/s^2 // radians
                accel_x = float(input_data['data']['accel_x'])
                accel_y = float(input_data['data']['accel_y'])
                accel_z = float(input_data['data']['accel_z'])
                gyro_x = float(input_data['data']['gyro_x'])
                gyro_y = float(input_data['data']['gyro_y'])
                gyro_z = float(input_data['data']['gyro_z'])
                timestamp = input_data['timestamp']
                validity = input_data['validity']

                self.debug_input_data(input_data, timestamp)

                # Delta time since last input
                dt = self.input_data_dt(timestamp)

                # TODO: compute position

                # TODO: remove gravity
                self.complementary_filter(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt)
                self.position_estimation_simple(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt)

                # TODO: transformation IMU Coordinates to Camera coordinates

            # TODO: evaluate position quality

            # TODO: save last few estimations with absolute timestamp

            # TODO: answer to position requests interpolating between saved data

            self.downsample_publish()

            # Time for processing
            if DEBUG_POSITION == 3:  # Full debug
                self.logger.info("Time needed for the loop : {} s. "
                                 .format((self.get_time_ns() - self.loop_time)/DIVIDER_OUTPUTS_SECONDS))

    # FUNCTION IMPLEMENTATION
    # dt time calculation between two consecutive analyzed samples. Input : ms, output : s
    def input_data_dt(self, timestamp):
        dt = (timestamp - self.timestamp) / 1000

        if DEBUG_POSITION > 2:
            self.logger.info("Time between elements : dt = {}. ".format(dt))
            self.logger.info("Timestamp, {}, self.timestamp : {}. ".format(timestamp, self.timestamp))

        if self.timestamp != 0:
            # Update
            self.timestamp = timestamp
            return dt  # seconds
        elif DEBUG_POSITION > 1:
            self.logger.warning("dt error: dt = {}.".format(dt))

        self.logger.warning("Corrupted calculation delta time, returning 10ms, saving timestamp")
        # Update
        self.timestamp = timestamp
        return 0.01

    def complementary_filter(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt):
        # Gyroscope data # °/s * s #TODO : check formula
        self.roll -= gyro_y * dt
        self.pitch += gyro_x * dt
        self.yaw += gyro_z * dt # No idea if right

        # Compensate for drift with accelerometer data if within Sensitivity [0 to 2 G]
        force_magnitude = abs(accel_x) + abs(accel_y) + abs(accel_z)
        if ACCEL_RANGE * 0.1 < force_magnitude < ACCEL_RANGE:
            # https://robotics.stackexchange.com/questions/4677/how-to-estimate-yaw-angle-from-tri-axis-accelerometer-and-gyroscope
            # Roll # RAD
            rollAcc = atan2(accel_x, accel_z) #TODO : check formula
            self.roll = self.roll * 0.98 + rollAcc * 0.02
            # Pitch # RAD
            pitchAcc = atan2(accel_y, accel_z)
            self.pitch = self.pitch * 0.98 + pitchAcc * 0.02
            # yaw # RAD
            yawAcc = atan2(accel_x, accel_y)
            self.yaw = self.yaw * 0.98 + yawAcc * 0.02

    def position_estimation_simple(self,
                                   accel_x:float, accel_y:float, accel_z:float,
                                   gyro_x:float, gyro_y:float, gyro_z:float,
                                   dt:float):
        if dt > 0:
            # TODO: Integrate the acceleration
            self.speed_x += accel_x * dt
            self.speed_y += accel_y * dt
            self.speed_z += accel_z * dt
            # Integrate to get the position
            # TODO: review: does the angle affect the calculations?
            self.pos_x += self.speed_x * dt
            self.pos_y += self.speed_y * dt
            self.pos_z += self.speed_z * dt

    def downsample_publish(self):
        # Downsample to POSITION_PUBLISH_FREQ (Hz) and publish
        if (self.loop_time - self.timestamp_last_output) * POSITION_PUBLISH_FREQ > DIVIDER_OUTPUTS_SECONDS:
            data_dict = {'pos_x': self.get_pos_x(),
                         'pos_y': self.get_pos_y(),
                         'pos_z': self.get_pos_z(),
                         'angle_x': self.get_angle_x(),
                         'angle_y': self.get_angle_y(),
                         'angle_z': self.get_angle_z()
                         }

            self.debug_downsample_publish(data_dict)

            # Publish with the timestamp of the last element received and update timestamp
            self.publish("position", data_dict, POS_VALIDITY_MS, self.timestamp)
            # Update
            self.timestamp_last_output = self.loop_time

    def get_pos_x(self):
        return self.pos_x

    def get_pos_y(self):
        return self.pos_y

    def get_pos_z(self):
        return self.pos_z

    def get_angle_x(self):
        return self.roll

    def get_angle_y(self):
        return self.pitch

    def get_angle_z(self):
        return self.yaw

    def get_timestamp(self):
        return self.timestamp

    def get_time_ms(self):
        # https://www.python.org/dev/peps/pep-0418/#time-monotonic
        return int(round(monotonic() * 1000))

    def get_time_ns(self):
        return int(round(perf_counter() * DIVIDER_OUTPUTS_SECONDS)) # nanoseconds

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

            if DEBUG_POSITION == 3:
                self.logger.info("Data :  {}".format(input_data))

    def debug_downsample_publish(self, data_dict):
        # DEBUG
        if DEBUG_POSITION > 1:
            # Counter of valid values
            self.count_outputs += 1

            # Log output for each element sent
            self.logger.info("Sent data N° {}, time between samples : {}. "
                             .format(self.count_outputs, self.loop_time - self.timestamp_last_output))

            if DEBUG_POSITION == 3:
                self.logger.info("Data :  {}".format(data_dict))

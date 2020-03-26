import io
import platform
import re
from queue import Queue
from time import sleep, monotonic, perf_counter
from pathlib import Path

from .utils import *
from ..module import Module
from ...utils import DEFAULT_DATASET

# Read data IMU
# Compute Position and save to array
# Answer to position requests (interpolating)

class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[("position", 10)],
                                            input_topics=["drivers_module:accelerations"], log_dir=log_dir)
        self.args = args

    def start(self):
        if DEBUG_POSITION == 1:
            #INFO : 4 ms to receive, 3 ms to send, 9 microS if nothing executed
            self.logger.info("Starting position_estimation_module...")

        # General inits
        self.count_valid_input = 0  # Number of inputs processed
        self.countall = 0           # Number of loops since start
        self.count_outputs = 0      # Number of elements published
        # Timestamps
        self.timestamp_last_input = 0                       # time last input received
        self.timestamp_last_output = self.get_time_ns()     # time last output published
        self.loop_time = self.get_time_ns()                 # time start of loop

        while(True):
            # Retrieve data
            input_data = self.get("drivers_module:accelerations")
            self.loop_time = self.get_time_ns()

            self.countall += 1

            if input_data:
                accel_x = input_data['data']['accel_x']
                accel_y = input_data['data']['accel_y']
                accel_z = input_data['data']['accel_z']
                gyro_x = input_data['data']['gyro_x']
                gyro_y = input_data['data']['gyro_y']
                gyro_z = input_data['data']['gyro_z']
                timestamp = input_data['timestamp']
                validity = input_data['validity']

                if DEBUG_POSITION > 1:
                    #Counter of valid values
                    self.count_valid_input += 1

                    #Difference in time from the timestamp of the data
                    if self.timestamp_last_input == 0:
                        timeDelta = 0
                    else:
                        timeDelta = timestamp - self.timestamp_last_input

                    #Log output for each element received
                    self.logger.info("Received Acceleration element N° {} from {}, time between samples : {}. "
                                     .format(self.count_valid_input, self.countall, timeDelta))
                    self.timestamp_last_input = timestamp

            # TODO: compute position

            # TODO: evaluate position quality

            # TODO: save last few estimations

            # TODO: answer to position requests

            # TODO: Downsample to POSITION_PUBLISH_FREQ (Hz) and publish
            if (self.loop_time - self.timestamp_last_output) * POSITION_PUBLISH_FREQ > 1000000000:
                data_dict = {'pos_x': self.get_pos_x(),
                             'pos_y': self.get_pos_y(),
                             'pos_z': self.get_pos_z(),
                             'angle_x': self.get_angle_x(),
                             'angle_y': self.get_angle_y(),
                             'angle_z': self.get_angle_z()
                             }
                timestamp = self.get_timestamp()

                # DEBUG
                if DEBUG_POSITION > 1:  # Full debug
                    # Counter of valid values
                    self.count_outputs += 1

                    # Log output for each element sent
                    self.logger.info("Sent data N° {}, time between samples : {}. "
                                     .format(self.count_outputs, self.loop_time - self.timestamp_last_output))

                # Publish and update timestamp
                self.publish("position", data_dict, POS_VALIDITY_MS, timestamp)
                self.timestamp_last_output = self.loop_time

            # Time for processing
            if DEBUG_POSITION > 2:  # Full debug
                self.logger.info("Time needed for the loop : {}. "
                                 .format(self.get_time_ns() - self.loop_time))

    # def myfunction(self):
    #     self.logger.warning("AX: {}, AY: {}, AZ: {}, GX: {}, GY: {}, GZ: {}".format(
    #         ACCEL_CALIB_X, ACCEL_CALIB_Y, ACCEL_CALIB_Z, GYRO_CALIB_X, GYRO_CALIB_Y, GYRO_CALIB_Z))

    # def ComplementaryFilter(short accData[3], short gyrData[3], float * pitch, float * roll):
    #     float pitchAcc, rollAcc;
    #
    #     # Integrate the gyroscope data -> int(angularSpeed) = angle
    #     * pitch += ((float)gyrData[0] / GYROSCOPE_SENSITIVITY) * dt; # Angle around the X - axis
    #     * roll -= ((float)gyrData[1] / GYROSCOPE_SENSITIVITY) * dt; # Angle around the Y - axis
    #
    #     # Compensate for drift with accelerometer data if !bullshit
    #     # Sensitivity = -2 to 2 G at 16Bit -> 2G = 32768 & & 0.5G = 8192
    #     int forceMagnitudeApprox = abs(accData[0]) + abs(accData[1]) + abs(accData[2]);
    #     if (forceMagnitudeApprox > 8192 & & forceMagnitudeApprox < 32768):
    #         # Turning around the X axis results in a vector on the Y-axis
    #         pitchAcc = atan2f((float)accData[1], (float)accData[2]) * 180 / M_PI;
    #         * pitch = * pitch * 0.98 + pitchAcc * 0.02;
    #
    #         # Turning around the Y axis results in a vector on the X-axis
    #         rollAcc = atan2f((float)accData[0], (float)accData[2]) * 180 / M_PI;
    #         * roll = * roll * 0.98 + rollAcc * 0.02;

    # TODO: function implementation
    def get_pos_x(self):
        return 0

    def get_pos_y(self):
        return 0

    def get_pos_z(self):
        return 0

    def get_angle_x(self):
        return 0

    def get_angle_y(self):
        return 0

    def get_angle_z(self):
        return 0

    def get_timestamp(self):
        return 0

    def get_time_ms(self):
        # https://www.python.org/dev/peps/pep-0418/#time-monotonic
        return int(round(monotonic() * 1000))

    def get_time_ns(self):
        return int(round(perf_counter() * 1000000000)) # nanoseconds
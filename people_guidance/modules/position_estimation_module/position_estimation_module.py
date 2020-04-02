import io
import platform
import re
import math
from time import sleep, monotonic, perf_counter
from scipy.spatial.transform import Rotation as R

from queue import Queue
from pathlib import Path

from .utils import *
from ..drivers_module import ACCEL_G
from ..module import Module
from ...utils import DEFAULT_DATASET

# Position estimation based on the data from the accelerometer and the gyroscope
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
        self.timestamp_last_output = self.get_time_ms()     # time last output published
        self.loop_time = self.get_time_ms()                 # time start of loop

        # Initialization and tracking
        self.dt_initialised = False
        self.last_data_dict_published = None

        # Output (m)
        self.pos_x = 0.
        self.pos_y = 0.
        self.pos_z = 0.
        # Output_speed (m/s)
        self.speed_x = 0.
        self.speed_y = 0.
        self.speed_z = 0.
        # Timestamp element
        self.timestamp:int = 0
        # Roll Pitch Yaw (Radians)
        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0.

    def start(self):
        if DEBUG_POSITION >= 1:
            self.logger.info("Starting position_estimation_module...")

        while(True):
            # Retrieve data
            input_data = self.get("drivers_module:accelerations")
            # sleep(0.1)

            if DEBUG_POSITION > 1:
                self.countall += 1  # count number of time the loop gets executed
                if DEBUG_POSITION >= 3:
                    self.loop_time = self.get_time_ms()
                    if DEBUG_POSITION == 4:
                        self.logger.info("loop time : {:.4f}".format(self.loop_time))

            # if DEBUG_POSITION >= 3:
            #     self.logger.info(input_data)

            if input_data: # m/s^2 // °/s
                accel_x = float(input_data['data']['accel_x'])
                accel_y = float(input_data['data']['accel_y'])
                accel_z = float(input_data['data']['accel_z'])
                gyro_x = float(input_data['data']['gyro_x']) * math.pi / 180
                gyro_y = float(input_data['data']['gyro_y']) * math.pi / 180
                gyro_z = float(input_data['data']['gyro_z']) * math.pi / 180
                timestamp = input_data['timestamp']
                validity = input_data['validity']

                self.debug_input_data(input_data, timestamp)

                # Delta time since last input
                dt = self.input_data_dt(timestamp)

                # TODO: compute position

                # TODO: remove gravity
                self.complementary_filter(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt)
                self.position_estimation_simple(accel_x, accel_y, accel_z, dt)

                # TODO: transformation IMU Coordinates to Camera coordinates

            # TODO: evaluate position quality

            # TODO: save last few estimations with absolute timestamp

            # Downsample to POSITION_PUBLISH_FREQ (Hz) and publish
            if (self.get_time_ms() - self.timestamp_last_output) * POSITION_PUBLISH_FREQ > 1:
                self.downsample_publish()

            # Time for processing
            if DEBUG_POSITION > 3:  # Full debug
                self.logger.info("Time needed for the loop : {:.4f} s. "
                                 .format((self.get_time_ms() - self.loop_time)))

    # FUNCTION IMPLEMENTATION
    # dt time calculation between two consecutive analyzed samples. Input : ms, output : s
    def input_data_dt(self, timestamp):
        dt = (timestamp - self.timestamp) / 1000

        if DEBUG_POSITION > 2:
            self.logger.info("Time between elements : dt = {}. ".format(dt))
            self.logger.info("Timestamp, {}, self.timestamp : {}. ".format(timestamp, self.timestamp))

        if self.timestamp != 0 and dt < 1 : # dt has not a too high value
            # Update
            self.timestamp = timestamp
            return dt  # seconds
        elif DEBUG_POSITION > 1:
            self.logger.info("dt error: dt = {}.".format(dt))

        if self.dt_initialised == True:
            self.logger.warning("Corrupted calculation delta time, returning 10ms, saving timestamp")
        else:
            self.dt_initialised = True
            self.logger.info("delta time error, returning 10ms, saving timestamp")
        # Update
        self.timestamp = timestamp
        return 0.01

    def complementary_filter(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt):
        # Only integrate for the yaw
        self.yaw += gyro_z * dt
        # Compensate for drift with accelerometer data
        roll_accel = math.atan(accel_y / math.sqrt(accel_x ** 2 + accel_z ** 2))
        pitch_accel = math.atan(accel_x / math.sqrt(accel_y ** 2 + accel_z ** 2))
        # Gyro data
        a = gyro_y * math.sin(self.roll) + gyro_z * math.cos(self.roll)
        roll_vel_gyro = gyro_x + a * math.tan(self.pitch)
        pitch_vel_gyro = gyro_y * math.cos(self.roll) - gyro_z * math.sin(self.roll)
        # Update estimation
        b = self.roll + roll_vel_gyro * dt
        self.roll = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * b + ALPHA_COMPLEMENTARY_FILTER * roll_accel
        c = self.pitch + pitch_vel_gyro * dt
        self.pitch = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * c + ALPHA_COMPLEMENTARY_FILTER * pitch_accel

    def position_estimation_simple(self,
                                   accel_x:float, accel_y:float, accel_z:float,
                                   dt:float):
        if dt > 0:
            # Integrate the acceleration after transforming the acceleration in world coordinates
            r = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=True)
            accel_rot = r.apply([accel_x, accel_y, accel_z])
            self.speed_x += accel_rot[0] * dt
            self.speed_y += accel_rot[1] * dt
            self.speed_z += (accel_rot[2] + ACCEL_G) * dt # TODO : check coordinate system setup
            # Integrate to get the position
            self.pos_x += self.speed_x * dt
            self.pos_y += self.speed_y * dt
            self.pos_z += self.speed_z * dt

    def track_values_attitude_estimation(self):
        # TODO: Build a loop to save the data
        # TODO: save ? the MA
        return

    def downsample_publish(self):
        data_dict = {'pos_x': self.get_pos_x(),
                     'pos_y': self.get_pos_y(),
                     'pos_z': self.get_pos_z(),
                     'angle_x': self.get_angle_x(),
                     'angle_y': self.get_angle_y(),
                     'angle_z': self.get_angle_z()
                     }
        if data_dict != self.last_data_dict_published:
            self.debug_downsample_publish(data_dict)

            # Publish with the timestamp of the last element received and update timestamp
            self.publish("position", data_dict, POS_VALIDITY_MS, self.timestamp)
            self.last_data_dict_published = data_dict
        # Update
        self.timestamp_last_output = self.get_time_ms()

    # TODO: answer to position requests interpolating between saved data

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
        return monotonic()

    def get_time_ns(self):
        return perf_counter() # nanoseconds

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
                             .format(self.count_outputs, self.get_time_ms() - self.timestamp_last_output))

            if DEBUG_POSITION >= 3:
                self.logger.info("Data :  {}".format(data_dict))

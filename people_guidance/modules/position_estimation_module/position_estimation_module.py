import io
import platform
import re
from queue import Queue
from time import sleep, monotonic
from pathlib import Path

# from .utils import *
from ..module import Module
from ...utils import DEFAULT_DATASET

# Read data IMU
# Compute Position and save to array
# Answer to position requests (interpolating)

class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[],
                                            input_topics=["drivers_module:accelerations"], log_dir=log_dir)
        self.args = args

    def start(self):
        self.logger.info("Starting position_estimation_module...")

        # General inits
        self.tmp = None

        while(True):
            # TODO: retrieve data
            sleep(1)
            input_data = self.get("drivers_module:accelerations")
            if input_data:
                self.logger.info(f"Received Acceleration with shape {input_data['data'].shape}. ")

            # # From DriverModule
            # data_dict = {'accel_x': self.get_accel_x(),
            #              'accel_y': self.get_accel_y(),
            #              'accel_z': self.get_accel_z(),
            #              'gyro_x': self.get_gyro_x(),
            #              'gyro_y': self.get_gyro_y(),
            #              'gyro_z': self.get_gyro_z()
            #              }
            # self.publish("accelerations", data_dict,
            #              IMU_VALIDITY_MS, timestamp)

            # TODO: compute position

            # TODO: evaluate position quality

            # TODO: save last few estimations

            # TODO: answer to position requests

    # def myfunction(self):
    #     self.logger.warning("AX: {}, AY: {}, AZ: {}, GX: {}, GY: {}, GZ: {}".format(
    #         ACCEL_CALIB_X, ACCEL_CALIB_Y, ACCEL_CALIB_Z, GYRO_CALIB_X, GYRO_CALIB_Y, GYRO_CALIB_Z))

    def cleanup(self):
        # any cleanup code (i.e. closing serial connections etc.) should be put here. This function is called even if an exception occurrs.
        pass
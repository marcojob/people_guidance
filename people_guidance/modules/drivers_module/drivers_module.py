import smbus
from pathlib import Path
from time import sleep

from ..module import Module
from .utils import *


class DriversModule(Module):
    def __init__(self, log_dir: Path):
        super(DriversModule, self).__init__(name="drivers_module", outputs=[],
                                            input_topics=[], log_dir=log_dir)

    def start(self):
        self.logger.info("Starting drivers")

        self.bus = smbus.SMBus(1)
        self.bus.write_byte_data(ADDR, PWR_MGMT_1, 0)

        self.set_accel_range()
        self.set_gyro_range()

        if DO_CALIB is True:
            self.imu_calibration()

        while(True):
            # Test prints
            self.logger.info("AX: {}, AY: {}, AZ: {}".format(
                self.get_accel_x(), self.get_accel_y(), self.get_accel_z()))
            sleep(1)

    def set_accel_range(self):
        # Get current config
        config = self.read_byte(ACCEL_CONFIG)
        # Datasheet chapter 4.4, set to +-2g range
        config &= ~(1 << 3)
        config &= ~(1 << 4)
        # Write back
        self.bus.write_byte_data(ADDR, ACCEL_CONFIG, config)

    def set_gyro_range(self):
        # Get current gyro config
        config = self.read_byte(GYRO_CONFIG)
        # Datasheet chapter 4.5, set to +-250°/s range
        config &= ~(1 << 3)
        config &= ~(1 << 4)
        # Write back
        self.bus.write_byte_data(ADDR, GYRO_CONFIG, config)

    def get_accel_x(self):
        return self.read_word_2c(ACCEL_XOUT_H_REG)*ACCEL_COEFF - ACCEL_CALIB_X

    def get_accel_y(self):
        return self.read_word_2c(ACCEL_YOUT_H_REG)*ACCEL_COEFF - ACCEL_CALIB_Y

    def get_accel_z(self):
        return self.read_word_2c(ACCEL_ZOUT_H_REG)*ACCEL_COEFF - ACCEL_CALIB_Z

    def get_gyro_x(self):
        return self.read_word_2c(GYRO_XOUT_H_REG)*GYRO_COEFF - GYRO_CALIB_X

    def get_gyro_y(self):
        return self.read_word_2c(GYRO_YOUT_H_REG)*GYRO_COEFF - GYRO_CALIB_Y

    def get_gyro_z(self):
        return self.read_word_2c(GYRO_ZOUT_H_REG)*GYRO_COEFF - GYRO_CALIB_Z

    def read_byte(self, reg):
        return self.bus.read_byte_data(ADDR, reg)

    def read_word(self, reg):
        h = self.bus.read_byte_data(ADDR, reg)
        l = self.bus.read_byte_data(ADDR, reg+1)
        value = (h << 8) + l
        return value

    def read_word_2c(self, reg):
        val = self.read_word(reg)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def imu_calibration(self, samples=100):
        self.logger.warning("IMU Calibration is starting now")
        ACCEL_CALIB_X = 0.0
        ACCEL_CALIB_Y = 0.0
        ACCEL_CALIB_Z = 0.0

        GYRO_CALIB_X = 0.0
        GYRO_CALIB_Y = 0.0
        GYRO_CALIB_Z = 0.0
        self.logger.warning("Calibrating X-axis")
        sleep(5)
        for s in range(samples):
            ACCEL_CALIB_X += self.get_accel_x() - ACCEL_G
            GYRO_CALIB_X += self.get_gyro_x()
            sleep(0.01)
        self.logger.warning("Calibrating Y-axis")
        sleep(5)
        for s in range(samples):
            ACCEL_CALIB_Y += self.get_accel_y() - ACCEL_G
            GYRO_CALIB_Y += self.get_gyro_y()
            sleep(0.01)
        self.logger.warning("Calibrating Z-axis")
        sleep(5)
        for s in range(samples):
            ACCEL_CALIB_Z += self.get_accel_z() + ACCEL_G
            GYRO_CALIB_Z += self.get_gyro_z()
            sleep(0.01)

        ACCEL_CALIB_X /= samples
        ACCEL_CALIB_Y /= samples
        ACCEL_CALIB_Z /= samples
        GYRO_CALIB_X /= samples
        GYRO_CALIB_Y /= samples
        GYRO_CALIB_Z /= samples

        self.logger.warning("AX: {}, AY: {}, AZ: {}, GX: {}, GY: {}, GZ: {}".format(
            ACCEL_CALIB_X, ACCEL_CALIB_Y, ACCEL_CALIB_Z, GYRO_CALIB_X, GYRO_CALIB_Y, GYRO_CALIB_Z))

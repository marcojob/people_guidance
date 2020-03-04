import smbus

from pathlib import Path
from time import sleep

from ..module import Module

power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c
address = 0x68


class DriversModule(Module):
    def __init__(self, log_dir: Path):
        super(DriversModule, self).__init__(name="drivers_module", outputs=[],
                                            input_topics=[], log_dir=log_dir)

    def start(self):
        self.logger.info("Starting drivers")

        self.bus = smbus.SMBus(1)
        self.bus.write_byte_data(address, power_mgmt_1, 0)

        while True:
            sleep(0.5)
            rot_x = self.read_word_2c(0x43)
            rot_y = self.read_word_2c(0x45)
            rot_z = self.read_word_2c(0x47)

            acc_x = self.read_word_2c(0x3b)
            acc_y = self.read_word_2c(0x3d)
            acc_z = self.read_word_2c(0x3f)
            self.logger.info("rot_x: {}, rot_y: {}, rot_z: {}, acc_x: {}, acc_y: {}, acc_z: {}".format(
                rot_x, rot_y, rot_z, acc_x, acc_y, acc_z))

    def read_byte(self, reg):
        return self.bus.read_byte_data(address, reg)

    def read_word(self, reg):
        h = self.bus.read_byte_data(address, reg)
        l = self.bus.read_byte_data(address, reg+1)
        value = (h << 8) + l
        return value

    def read_word_2c(self, reg):
        val = self.read_word(reg)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

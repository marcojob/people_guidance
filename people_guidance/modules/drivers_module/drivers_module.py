import io
import platform
import re
import cv2
import numpy as np
from queue import Queue
from time import sleep, monotonic
from statistics import median
from pathlib import Path

from .utils import *
from ..module import Module
from ...utils import DEFAULT_DATASET

if platform.uname().machine == 'armv7l':
    RPI = True
else:
    RPI = False

if RPI:
    import smbus
    from picamera import mmal, mmalobj


class DriversModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(DriversModule, self).__init__(name="drivers_module",
                                            outputs=[("images", 10000),
                                                     ("accelerations", 100), ("accelerations_vis", 100)],
                                            inputs=[], log_dir=log_dir)
        self.args = args

    def start(self):
        self.logger.info("Starting drivers")

        # General inits
        self.files_dir = None
        self.imu_data = None
        self.img_data = None
        self.img_counter = 0
        self.RECORD_MODE = False
        self.REPLAY_MODE = False

        self.data_window = {topic: list() for topic in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']}

        # IMU INITS
        self.imu_next_sample_ms = self.get_time_ms()

        if RPI:
            # CAMERA INITS
            self.camera = mmalobj.MMALCamera()
            self.encoder = mmalobj.MMALImageEncoder()

            # Queues
            self.q_img = Queue()

            # CAMERA SETUP
            self.camera_pipeline_setup()
            self.camera_start()

            # IMU INITS
            self.bus = smbus.SMBus(1)

            # IMU SETUP
            self.bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
            self.set_accel_range()
            self.set_gyro_range()

        # Get hardware configuration mode
        self.setup_hardware_configuration()

        while True:
            # Not replay mode, either normal or record mode
            if not self.REPLAY_MODE:
                # IMU gets sampled at a fixed frequency
                if self.get_time_ms() > self.imu_next_sample_ms:
                    timestamp = self.get_time_ms()

                    # Schedule the next sample time
                    self.imu_next_sample_ms = timestamp + IMU_SAMPLE_TIME_MS

                    # Dict of IMU data
                    data_dict = {'accel_x': self.get_accel_x(),
                                 'accel_y': self.get_accel_y(),
                                 'accel_z': self.get_accel_z(),
                                 'gyro_x': self.get_gyro_x(),
                                 'gyro_y': self.get_gyro_y(),
                                 'gyro_z': self.get_gyro_z(),
                                 "timestamp": timestamp
                                 }

                    # Track window for median filter
                    data_dict = self.track_val_median_filter(data_dict)

                    if self.RECORD_MODE:
                        # In record mode, we want to write data into the open file
                        self.imu_data.write(f"{timestamp}: " +
                                            f"accel_x: {data_dict['accel_x']}, " +
                                            f"accel_y: {data_dict['accel_y']}, " +
                                            f"accel_z: {data_dict['accel_z']}, " +
                                            f"gyro_x: {data_dict['gyro_x']}, " +
                                            f"gyro_y: {data_dict['gyro_y']}, " +
                                            f"gyro_z: {data_dict['gyro_z']}\n")
                        self.imu_data.flush()
                    else:
                        # In normal mode, we just publish the data
                        self.publish("accelerations", data_dict, IMU_VALIDITY_MS)
                        self.publish("accelerations_vis", data_dict, -1)
            else:

                # We are in replay mode
                if not self.imu_timestamp:
                    # We read one line of data
                    imu_str = self.imu_data.readline()

                    # Look for data in the right format
                    out = re.search(IMU_RE_MASK, imu_str)
                    if out:
                        # Find timestamp of data
                        self.imu_timestamp = int(out.group(1))

                        if not self.imu_first_timestamp:
                            self.imu_first_timestamp = self.imu_timestamp

                        # Populate dict with data, as if it was sampled normally
                        self.imu_data_dict = {'accel_x': float(out.group(2)),
                                              'accel_y': float(out.group(3)),
                                              'accel_z': float(out.group(4)),
                                              'gyro_x': float(out.group(5)),
                                              'gyro_y': float(out.group(6)),
                                              'gyro_z': float(out.group(7)),
                                              "timestamp": int(out.group(1))
                                              }

                        # Track window for median filter
                        self.imu_data_dict = self.track_val_median_filter(self.imu_data_dict)

                # If the relative time is correct, we publish the data

                if self.imu_timestamp and self.get_time_ms() - self.replay_start_timestamp > self.imu_timestamp - self.imu_first_timestamp:
                    self.publish("accelerations", self.imu_data_dict, IMU_VALIDITY_MS)
                    self.publish("accelerations_vis", self.imu_data_dict, -1)

                    # Reset the timestamp so that a new dataset is read
                    self.imu_timestamp = None

            # We want to forward image data as fast and often as possible
            # Not replay mode, either normal or record mode
            if not self.REPLAY_MODE:
                if not self.q_img.empty():
                    # Get next img from queue
                    data_dict = self.q_img.get()
                    timestamp = data_dict['timestamp']

                    if self.RECORD_MODE:
                        self.img_counter += 1
                        # Keep track of image counter and timestamp
                        self.img_data.write(f"{self.img_counter}: {timestamp}\n")
                        self.img_data.flush()

                        # Write image
                        img = self.files_dir / 'imgs' / f"img_{self.img_counter:04d}.jpg"
                        img_f = io.open(img, 'wb')
                        img_f.write(data_dict['data'])
                        img_f.close()
                    else:
                        # In normal mode we just publish the image
                        self.publish("images", data_dict, IMAGES_VALIDITY_MS)
            else:
                # We are in replay mode
                if not self.img_timestamp:
                    # Read from the file that keeps track of timestamps
                    img_str = self.img_data.readline()

                    if img_str:
                        out = re.search(r'([0-9]*): ([0-9]*)', img_str)
                        if out:
                            self.img_timestamp = int(out.group(2))

                            if not self.img_first_timestamp:
                                self.img_first_timestamp = self.img_timestamp

                            # Read the image corresponding to the counter and timestamp
                            img_filename = f"img_{int(out.group(1)):04d}.jpg"
                            img_file_path = self.files_dir / 'imgs' / img_filename

                            with open(img_file_path, 'rb') as fp:
                                img_data_file = fp.read()

                            # Decode image
                            self.img = cv2.imdecode(np.frombuffer(img_data_file, dtype=np.int8), flags=cv2.IMREAD_COLOR)

                            # Undistort image
                            if UNDISTORT_IMAGE:
                                self.img = cv2.undistort(self.img, self.intrinsic_matrix, self.distortion_coeffs)

                            # Resize image
                            if RESIZE_IMAGE:
                                self.img = cv2.resize(self.img, RESIZED_IMAGE)


                # If the relative time is correct, we publish the data

                if self.img_timestamp and self.get_time_ms() - self.replay_start_timestamp > self.img_timestamp - self.img_first_timestamp:
                    self.publish("images", {"data": self.img, "timestamp": self.img_timestamp}, -1)

                    # Reset the timestamp so that a new dataset is read
                    self.img_timestamp = None

    def camera_pipeline_setup(self):
        # Camera output setup
        self.camera.outputs[0].format = mmal.MMAL_ENCODING_RGB24
        self.camera.outputs[0].framesize = CAMERA_FRAMESIZE
        self.camera.outputs[0].framerate = CAMERA_FRAMERATE
        self.camera.outputs[0].params[mmal.MMAL_PARAMETER_ROTATION] = CAMERA_ROTATION
        self.camera.outputs[0].commit()

        # Encoder input setup
        self.encoder.inputs[0].format = mmal.MMAL_ENCODING_RGB24
        self.encoder.inputs[0].framesize = CAMERA_FRAMESIZE
        self.encoder.inputs[0].commit()

        # Encoder output setup
        self.encoder.outputs[0].copy_from(self.encoder.inputs[0])
        self.encoder.outputs[0].format = mmal.MMAL_ENCODING_JPEG
        self.encoder.outputs[0].params[mmal.MMAL_PARAMETER_JPEG_Q_FACTOR] = CAMERA_JPEG_QUALITY
        self.encoder.outputs[0].commit()

        # Connect main encoder input to camera output
        self.encoder.connect(self.camera.outputs[0])
        self.encoder.connection.enable()

        # Rename main output
        self.main_output = self.encoder.outputs[0]

    def image_callback(self, port, buf):
        # Is called in separate thread
        self.q_img.put({'data': buf.data,
                        'timestamp': self.get_time_ms()})
        return False

    def get_time_ms(self):
        # https://www.python.org/dev/peps/pep-0418/#time-monotonic
        return int(round(monotonic() * 1000))

    def camera_start(self):
        self.main_output.enable(self.image_callback)

    def camera_stop(self):
        self.encoder.connection.disable()

    def track_val_median_filter(self, data_dict):
        for key in self.data_window.keys():
            # If longer than median filter window size, delete
            if len(self.data_window[key]) > LEN_MEDIAN:
                del self.data_window[key][0]

            # Append new data
            self.data_window[key].append(data_dict[key])

            # Get median value
            data_dict[key] = median(self.data_window[key])

        return data_dict


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

    def setup_hardware_configuration(self):
        # Cannot replay and record at the same time

        if len([arg for arg in (self.args.replay, self.args.record, self.args.deploy) if arg]) > 1:
            self.logger.error(
                "arguments replay, record and deploy are mutually exclusive. exiting...")
            exit()

        if self.args.record:
            self.RECORD_MODE = True
            self.files_dir = Path(self.args.record)

            # IMU files
            self.imu_data = (self.files_dir / 'imu_data.txt').open(mode='w')

            # Camera files
            self.img_data = (self.files_dir / 'img_data.txt').open(mode='w')
            (self.files_dir / 'imgs').mkdir(parents=True, exist_ok=True)

        elif self.args.deploy:
            assert RPI, f"The drivers_module can only be deployed to a Raspberry PI. Please choose the --replay flag"

        else:
            # Replay Mode is the default.
            self.REPLAY_MODE = True

            self.files_dir: Path

            if self.args.replay:
                self.files_dir = Path(self.args.replay)
                self.logger.info("Replaying dataset from {self.files_dir}")
            else:
                self.logger.info(f"No arguments specified. Will replay default dataset from {DEFAULT_DATASET}")
                if not DEFAULT_DATASET.is_dir():
                    raise FileNotFoundError(f"Could not find a dataset at the default location {DEFAULT_DATASET}. "
                                            f"Please move your dataset to the correct location or specify a custom location"
                                            f" dataset using the --replay command line argument.")

                self.files_dir = DEFAULT_DATASET

            self.replay_start_timestamp = self.get_time_ms()

            # IMU inits
            self.imu_data = (self.files_dir / 'imu_data.txt').open(mode='r')

            self.imu_data_dict = None
            self.imu_timestamp = None
            self.imu_first_timestamp = None

            # Camera inits
            self.img_data = (self.files_dir / 'img_data.txt').open(mode='r')
            self.img_data_file = None
            self.img_timestamp = None
            self.img_first_timestamp = None

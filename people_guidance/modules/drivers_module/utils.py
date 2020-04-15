# MPU 6050 Registers,
# datasheet https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf

# FLAGS
DO_CALIB = False

# CAMERA PARAMETERS
CAMERA_FRAMERATE = 40
CAMERA_FRAMESIZE = (1640, 1232)
CAMERA_JPEG_QUALITY = 90
CAMERA_ROTATION = 180

IMAGES_VALIDITY_MS = 1000.0/CAMERA_FRAMERATE * 1.10

# IMU PARAMETERS
ADDR = 0x68

ACCEL_CONFIG = 0x17

GYRO_CONFIG = 0x18

ACCEL_XOUT_H_REG = 0x3B
ACCEL_XOUT_L_REG = 0x3C

ACCEL_YOUT_H_REG = 0x3D
ACCEL_YOUT_L_REG = 0x3E

ACCEL_ZOUT_H_REG = 0x3F
ACCEL_ZOUT_L_REG = 0x40

GYRO_XOUT_H_REG = 0x43
GYRO_XOUT_L_REG = 0x44

GYRO_YOUT_H_REG = 0x45
GYRO_YOUT_L_REG = 0x46

GYRO_ZOUT_H_REG = 0x47
GYRO_ZOUT_L_REG = 0x48

PWR_MGMT_1 = 0x6b
PWR_MGMT_2 = 0x6c

IMU_SAMPLE_FREQ_HZ = 100
IMU_SAMPLE_TIME_MS = 1000.0*1.0/IMU_SAMPLE_FREQ_HZ

IMU_VALIDITY_MS = IMU_SAMPLE_TIME_MS * 1.10

# Earth acceleration in Zurich
ACCEL_G = -9.80600

# Accel full range in m/s^2
ACCEL_RANGE = 2.0*ACCEL_G

# 16 signed, max val
IMU_MAX_VAL = 32768

# Gyro full range in °/s
GYRO_RANGE = 250

# Coeffs
ACCEL_COEFF = ACCEL_RANGE/IMU_MAX_VAL
GYRO_COEFF = GYRO_RANGE/IMU_MAX_VAL

# Calibrations values
ACCEL_CALIB_X = 0
ACCEL_CALIB_Y = 0
ACCEL_CALIB_Z = 0

GYRO_CALIB_X = 0
GYRO_CALIB_Y = 0
GYRO_CALIB_Z = 0

IMU_RE_MASK = r'([0-9]*): accel_x: ([0-9.-]*), ' + \
    'accel_y: ([0-9.-]*), ' + \
    'accel_z: ([0-9.-]*), ' + \
    'gyro_x: ([0-9.-]*), ' + \
    'gyro_y: ([0-9.-]*), ' + \
    'gyro_z: ([0-9.-]*)'

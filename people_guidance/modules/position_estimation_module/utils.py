import collections
import platform
import random

import numpy as np
import matplotlib.pyplot as plt


IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])


# Debug mode
DEBUG_POSITION = 0  # 0: none, 1: light, 2: mid, 3: extended, 4: full debug options
VISUALIZE_LOCALLY = True
# Queue output
POS_VALIDITY_MS = 100
POSITION_PUBLISH_FREQ = 1
POSITION_PUBLISH_ACC_FREQ = 0
POSITION_PUBLISH_INPUT_FREQ = 0

# Requests
TRACK_FOR_REQUEST_POSITION_NUMBER_ELT_KEEP = 1000

# Reduce the velocity to reduce drift
METHOD_RESET_VELOCITY = False
RESET_VEL_FREQ = 200 # select value above 100 to compensate after each step  TODO : prone to dt
RESET_VEL_FREQ_COEF_X = 0.991
RESET_VEL_FREQ_COEF_Y = 0.991
RESET_VEL_FREQ_COEF_Z = 0.97

# Error calculation
MEASURE_SUMMED_ERROR_ACC = False
METHOD_ERROR_ACC_CORRECTION = False # True, worse otherwise
PUBLISH_SUMMED_MEASURE_ERROR_ACC = 4

# dataset_1
SUM_DT_DATASET_1 = 9.622
SUM_ELT_DATASET_1 = 2751
SUM_ACC_DATASET_1 = [-42.77572310263783, -170.8575616629757, -1633.801397779215]

if METHOD_ERROR_ACC_CORRECTION:
    CORRECTION_ACC = np.divide(SUM_ACC_DATASET_1, SUM_ELT_DATASET_1)
else:
    CORRECTION_ACC = [0, 0, 0]

# Time calculation to get an output in seconds
DIVIDER_OUTPUTS_SECONDS = 1000000000
DIVIDER_OUTPUTS_mSECONDS = 1000

# Complementary filter parameter
ALPHA_COMPLEMENTARY_FILTER = 0.02


def visualize_locally(pos, frame: IMUFrame, plot_acc: bool = False, plot_angles: bool = False):
    """

    :param pos: the position to visualize
    :param frame: the IMUFrame to visualize
    :param plot_acc: should accelerations be plotted?
    :param plot_angles: should angles be plotted?
    :param dropout: the percentage of calls to this function that will actually update the output
    :return:
    """
    # place figure in top left corner
    plt.figure(1, figsize=(10, 5))
    if platform.system() != "Windows":
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(5, 5, 1000, 1000)

    plt.subplot(2, 3, 1)
    plt.scatter(pos.x, pos.y)
    plt.title('Position estimation')
    plt.suptitle(f'Parameters : \n'
                 f'METHOD_RESET_VELOCITY: {METHOD_RESET_VELOCITY}, RESET_VEL_FREQ : {RESET_VEL_FREQ}, \n'
                 f'RESET_VEL_FREQ_COEF_X : {RESET_VEL_FREQ_COEF_X}, RESET_VEL_FREQ_COEF_Y : {RESET_VEL_FREQ_COEF_Y}, RESET_VEL_FREQ_COEF_Z : {RESET_VEL_FREQ_COEF_Z}, \n'
                 f'METHOD_ERROR_ACC_CORRECTION : {METHOD_ERROR_ACC_CORRECTION}, CORRECTION_ACC : {CORRECTION_ACC}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')



    plt.subplot(2, 3, 2)
    plt.scatter(pos.x, pos.z)
    plt.title('Position estimation')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')

    if plot_angles:
        plt.subplot(2, 3, 3)
        plt.scatter(pos.roll, pos.pitch)
        plt.title('Angle')
        plt.xlabel('roll [rad]')
        plt.ylabel('pitch [rad]')

        plt.subplot(2, 3, 4)
        plt.scatter(pos.roll, pos.yaw)
        plt.title('Angle')
        plt.xlabel('roll [rad]')
        plt.ylabel('yaw [rad]')

    if plot_acc:
        plt.subplot(2, 3, 5)
        plt.scatter(frame.ax, frame.ay)
        plt.title('Acceleration')
        plt.xlabel('x [acc]')
        plt.ylabel('y [acc]')

        plt.subplot(2, 3, 6)
        plt.scatter(frame.ax, frame.az)
        plt.title('Acceleration')
        plt.xlabel('x [acc]')
        plt.ylabel('z [acc]')

    plt.pause(0.0001)

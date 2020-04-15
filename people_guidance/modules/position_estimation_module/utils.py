import collections
import platform
import random

import numpy as np
import matplotlib.pyplot as plt


IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])


# Debug mode
VISUALIZE_LOCALLY = False

# Queue output
POS_VALIDITY_MS = 100
# Debug output frequency. A value above 99 means that new elements are always published
POSITION_PUBLISH_NEW_TIMESTAMP = 0
POSITION_PUBLISH_FREQ = 1
POSITION_PUBLISH_ACC_FREQ = 1
POSITION_PUBLISH_INPUT_FREQ = 1

# Requests
TRACK_FOR_REQUEST_POSITION_NUMBER_ELT_KEEP = 1000

# Complementary filter parameter
ALPHA_COMPLEMENTARY_FILTER = 0.1

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


def visualize_locally(pos, frame: IMUFrame, drift_tracking, acceleration, plot_pos: bool = True, plot_angles: bool = False,
                      plot_acc_input: bool = False, plot_acc_transformed: bool = False):
    """
    :param pos: the position to visualize
    :param frame: the IMUFrame to visualize
    :param plot_pos: should the position be plotted?
    :param plot_acc: should accelerations be plotted?
    :param plot_angles: should angles be plotted?
    :param dropout: the percentage of calls to this function that will actually update the output
    :return:
    """
    if (plot_pos or plot_acc or plot_angles) and VISUALIZE_LOCALLY:
        # place figure in top left corner
        plt.figure(1, figsize=(12, 15))
        plt.tight_layout()
        if platform.system() != "Windows":
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(5, 5, 1000, 1000)

        # responsive window size and subplot init
        responsive_shift = 0 # Number of plots to be shown
        responsive_window_size_col = 0 # Number of columns to display
        responsive_shift_acc = 0 # adding another column
        if plot_pos:
            responsive_shift += 2
            responsive_window_size_col += 1
        if plot_angles:
            responsive_shift += 2
            responsive_window_size_col += 1
        if plot_acc_input:
            responsive_window_size_col += 1
            responsive_shift_acc = 2
        if plot_acc_transformed:
            responsive_window_size_col += 1

        correction_acc_display = CORRECTION_ACC
        if MEASURE_SUMMED_ERROR_ACC:
            # Display calculated correction
            correction_acc_display = [round(drift_tracking["total_acc_x"] / drift_tracking["n_elt_summed"], 3),
                                      round(drift_tracking["total_acc_y"] / drift_tracking["n_elt_summed"], 3),
                                      round(drift_tracking["total_acc_y"] / drift_tracking["n_elt_summed"], 3),
                                      drift_tracking["n_elt_summed"]]

        plt.suptitle(f'Parameters : METHOD_RESET_VELOCITY: {METHOD_RESET_VELOCITY}, RESET_VEL_FREQ : {RESET_VEL_FREQ}, '
                     f'RESET_VEL_FREQ_COEF_X : {RESET_VEL_FREQ_COEF_X}, RESET_VEL_FREQ_COEF_Y : {RESET_VEL_FREQ_COEF_Y}, '
                     f'RESET_VEL_FREQ_COEF_Z : {RESET_VEL_FREQ_COEF_Z}, '
                     f'METHOD_ERROR_ACC_CORRECTION : {METHOD_ERROR_ACC_CORRECTION}, '
                     f'MEASURE_SUMMED_ERROR_ACC : {MEASURE_SUMMED_ERROR_ACC}, CORRECTION_ACC : {correction_acc_display}',
                     x=0.001, y=.999, horizontalalignment='left', verticalalignment='top', fontsize = 5)

        if plot_pos:
            plt.subplot(2, responsive_window_size_col, 1)
            plt.scatter(pos.x, pos.y)
            plt.title('Position estimation')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')

            plt.subplot(2, responsive_window_size_col, 2)
            plt.scatter(pos.x, pos.z)
            plt.title('Position estimation')
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')

        if plot_angles:
            plt.subplot(2, responsive_window_size_col, responsive_shift - 1)
            plt.scatter(pos.roll, pos.pitch)
            plt.title('Angle')
            plt.xlabel('roll [rad]')
            plt.ylabel('pitch [rad]')

            plt.subplot(2, responsive_window_size_col, responsive_shift)
            plt.scatter(pos.roll, pos.yaw)
            plt.title('Angle')
            plt.xlabel('roll [rad]')
            plt.ylabel('yaw [rad]')

        if plot_acc_input:
            plt.subplot(2, responsive_window_size_col, responsive_shift + 1)
            plt.scatter(frame.ax, frame.ay)
            plt.title('Acceleration IN')
            plt.xlabel('x [acc]')
            plt.ylabel('y [acc]')

            plt.subplot(2, responsive_window_size_col, responsive_shift + 2)
            plt.scatter(frame.ax, frame.az)
            plt.title('Acceleration IN')
            plt.xlabel('x [acc]')
            plt.ylabel('z [acc]')

        if plot_acc_transformed:
            plt.subplot(2, responsive_window_size_col, responsive_shift + responsive_shift_acc + 1)
            plt.scatter(acceleration['x'], acceleration['y'])
            plt.title('Acceleration OUT')
            plt.xlabel('x [acc]')
            plt.ylabel('y [acc]')

            plt.subplot(2, responsive_window_size_col, responsive_shift + responsive_shift_acc + 2)
            plt.scatter(acceleration['x'], acceleration['z'])
            plt.title('Acceleration OUT')
            plt.xlabel('x [acc]')
            plt.ylabel('z [acc]')

        plt.pause(0.0001)

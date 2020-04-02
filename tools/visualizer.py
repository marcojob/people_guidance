# Visualizer tool

import threading
import time
import cv2
import re
import io
import socket
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from time import sleep

HOST = ""  # Host IP
PORT = 65432  # Port
TOPIC_LIST = ["pos_x", "pos_y", "pos_z", "angle_x", "angle_y", "angle_z", "preview"]  # All topics


# Dictionary for all data
data_dict = {topic: [0] for topic in TOPIC_LIST}
DATA_MAX_LEN = 500

KEYS = ["preview", "pos"]

ax_list = dict()
scatter_p = None
preview_p = None

POS_RE_MASK = r'([0-9]*): '+ \
    'pos_x: ([0-9.-]*), ' + \
    'pos_y: ([0-9.-]*), ' + \
    'pos_z: ([0-9.-]*), ' + \
    'angle_x: ([0-9.-]*), ' + \
    'angle_y: ([0-9.-]*), ' + \
    'angle_z: ([0-9.-]*)\n'


def animate_pos():
    global scatter_p
    global line_x, line_y, line_z

    # Current position and angles
    pos_x = data_dict["pos_x"][-1]
    pos_y = data_dict["pos_y"][-1]
    pos_z = data_dict["pos_z"][-1]
    angle_x = data_dict["angle_x"][-1]*np.pi/180.0
    angle_y = data_dict["angle_y"][-1]*np.pi/180.0
    angle_z = data_dict["angle_z"][-1]*np.pi/180.0
    r = R.from_rotvec(np.array([angle_x, angle_y, angle_z])).as_matrix()
    sc = 1

    if scatter_p == None:
        ax_list["pos"].set_title("pos")
        ax_list["pos"].set_xlim((-100, 20))
        ax_list["pos"].set_ylim((-100, 100))

        scatter_p = ax_list["pos"].scatter(
            data_dict["pos_x"], data_dict["pos_y"], data_dict["pos_z"])

        line_x = ax_list["pos"].plot([pos_x, sc*r[0][0]], [pos_y, sc*r[0][1]], [pos_z, sc*r[0][2]])
        line_y = ax_list["pos"].plot([pos_x, sc*r[1][0]], [pos_y, sc*r[1][1]], [pos_z, sc*r[1][2]])
        line_z = ax_list["pos"].plot([pos_x, sc*r[2][0]], [pos_y, sc*r[2][1]], [pos_z, sc*r[2][2]])

        ax_list["pos"].figure.canvas.draw()
    else:
        scatter_p._offsets3d = (data_dict["pos_x"], data_dict["pos_y"], data_dict["pos_z"])

        line_x[0].set_xdata([pos_x, r[0][0]])
        line_x[0].set_ydata([pos_y, r[0][1]])
        line_x[0].set_3d_properties([pos_z, r[0][2]])

        line_y[0].set_xdata([pos_x, r[1][0]])
        line_y[0].set_ydata([pos_y, r[1][1]])
        line_y[0].set_3d_properties([pos_z, r[1][2]])

        line_z[0].set_xdata([pos_x, r[2][0]])
        line_z[0].set_ydata([pos_y, r[2][1]])
        line_z[0].set_3d_properties([pos_z, r[2][2]])


def animate_preview():
    global preview_p

    if preview_p == None:
        ax_list["preview"].set_title("preview")
        ax_list["preview"].set_axis_off()

        preview_p = ax_list["preview"].imshow(data_dict["preview"][0])

        ax_list["preview"].figure.canvas.draw()
    else:
        preview_p.set_data(data_dict["preview"][0])
        ax_list["preview"].figure.canvas.draw()

def get_time_ms():
    # https://www.python.org/dev/peps/pep-0418/#time-monotonic
    return int(round(time.monotonic() * 1000))


def plot_main():
    fig = plt.figure()
    ax_list["preview"] = fig.add_subplot(1, 2, 1)
    ax_list["preview"].set_title("preview")
    ax_list["preview"].set_axis_off()

    ax_list["pos"] = fig.add_subplot(1, 2, 2, projection='3d')
    ax_list["pos"].set_title("pos")
    plt.show()


def socket_main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            while True:
                data_id = conn.recv(1)
                data_id_int = int.from_bytes(data_id, byteorder='little')

                # Data ID 0 is preview
                if data_id_int == 0:
                    data_len = conn.recv(4)
                    data_len_int = int.from_bytes(data_len, byteorder='little')

                    if data_len_int > 0:
                        buf = conn.recv(data_len_int)

                        try:
                            img_dec = cv2.imdecode(np.frombuffer(
                                buf, dtype=np.int8), flags=cv2.IMREAD_COLOR)

                            data_dict["preview"][0] = img_dec
                            animate_preview()
                        except Exception as e:
                            pass
                            # print(f"preview: {e}")

                elif data_id_int == 1:
                    data_len = conn.recv(4)
                    data_len_int = int.from_bytes(data_len, byteorder='little')

                    if data_len_int > 0:
                        buf = conn.recv(data_len_int)

                        try:
                            pos_data = np.frombuffer(buf, dtype=np.float32)
                            data_dict["pos_x"].append(-1.0*pos_data[0])
                            data_dict["pos_y"].append(-1.0*pos_data[1])
                            data_dict["pos_z"].append(0)
                            data_dict["angle_x"].append(pos_data[3])
                            data_dict["angle_y"].append(pos_data[4])
                            data_dict["angle_z"].append(pos_data[5])

                            animate_pos()

                        except Exception as e:
                            pass
                            # print(f"pos: {e}")


def replay_main(args):
    files_dir = Path(args.replay)
    preview_data = (files_dir / 'vis_data.txt').open(mode='r')
    pos_data = (files_dir / 'pos_data.txt').open(mode='r')

    img_timestamp = None
    img_first_timestamp = None
    replay_start_timestamp = get_time_ms()

    pos_timestamp = None
    pos_first_timestamp = None

    # We need give plot_main a bit of time
    sleep(1)

    while(True):
        if not img_timestamp:
            # Read from the file that keeps track of timestamps
            img_str = preview_data.readline()

            out = re.search(r'([0-9]*): ([0-9]*)', img_str)
            if out:
                img_timestamp = int(out.group(2))

                if not img_first_timestamp:
                    img_first_timestamp = img_timestamp

                # Read the image corresponding to the counter and timestamp
                img_filename = f"img_{int(out.group(1)):04d}.jpg"
                img_file_path = files_dir / 'vis' / img_filename

                img_f = io.open(img_file_path, 'rb')
                img_data_file = img_f.read()
                img_f.close()

        # If the relative time is correct, we publish the data
        if img_timestamp and get_time_ms() - replay_start_timestamp > \
                img_timestamp - img_first_timestamp:

            img_dec = cv2.imdecode(np.frombuffer(
                img_data_file, dtype=np.int8), flags=cv2.IMREAD_COLOR)
            data_dict["preview"][0] = img_dec
            animate_preview()
            img_timestamp = None

        if not pos_timestamp:
            # We read one line of data
            pos_str = pos_data.readline()

            # Look for data in the right format
            out = re.search(POS_RE_MASK, pos_str)
            if out:
                # Find timestamp of data
                pos_timestamp = int(out.group(1))

                if not pos_first_timestamp:
                    pos_first_timestamp = pos_timestamp

                # Populate dict with data, as if it was sampled normally
                pos_data_dict = {'pos_x': float(out.group(2)),
                                 'pos_y': float(out.group(3)),
                                 'pos_z': float(out.group(4)),
                                 'angle_x': float(out.group(5)),
                                 'angle_y': float(out.group(6)),
                                 'angle_z': float(out.group(7))}

        # If the relative time is correct, we publish the data
        if pos_timestamp and get_time_ms() - replay_start_timestamp > \
                pos_timestamp - pos_first_timestamp:

            data_dict["pos_x"].append(-1.0*pos_data_dict["pos_x"])
            data_dict["pos_y"].append(-1.0*pos_data_dict["pos_y"])
            data_dict["pos_z"].append(0)
            data_dict["angle_x"].append(pos_data_dict["angle_x"])
            data_dict["angle_y"].append(pos_data_dict["angle_y"])
            data_dict["angle_z"].append(pos_data_dict["angle_z"])

            animate_pos()

            pos_timestamp = None


def main():
    parser = ArgumentParser()
    parser.add_argument('--replay', '-p',
                        help='Path of folder where to replay dataset from',
                        type=str,
                        default='')

    args = parser.parse_args()

    if not args.replay:
        socket_thread = threading.Thread(target=socket_main)
        socket_thread.start()
    else:
        replay_thread = threading.Thread(target=replay_main, args=(args,))
        replay_thread.start()

    plot_main()


if __name__ == '__main__':
    main()

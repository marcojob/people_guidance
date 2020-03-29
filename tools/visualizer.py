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
from time import sleep

HOST = ""  # Host IP
PORT = 65432  # Port
TOPIC_LIST = ["pos_x", "pos_y", "pos_z", "preview"]  # All topics


# Dictionary for all data
data_dict = {topic: [0] for topic in TOPIC_LIST}
DATA_MAX_LEN = 500

KEYS = ["preview", "pos"]

ax_list = dict()


POS_RE_MASK = r'([0-9]*): pos_x: ([0-9.-]*), ' + \
    'pos_y: ([0-9.-]*), ' + \
    'pos_z: ([0-9.-]*)'


def animate_pos():
    ax_list["pos"].clear()
    ax_list["pos"].set_title("pos")
    ax_list["pos"].scatter(
        data_dict["pos_x"], data_dict["pos_y"], data_dict["pos_z"])
    ax_list["pos"].figure.canvas.draw()


def animate_preview():
    ax_list["preview"].clear()
    ax_list["preview"].set_title("preview")
    ax_list["preview"].set_axis_off()
    ax_list["preview"].imshow(data_dict["preview"][0])
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
                            print(e)

                elif data_id_int == 1:
                    data_len = conn.recv(4)
                    data_len_int = int.from_bytes(data_len, byteorder='little')

                    if data_len_int > 0:
                        buf = conn.recv(data_len_int)

                        try:
                            pos_data = np.frombuffer(buf, dtype=np.float32)
                            data_dict["pos_x"].append(pos_data[0])
                            data_dict["pos_y"].append(pos_data[1])
                            data_dict["pos_z"].append(pos_data[2])
                            animate_pos()

                        except Exception as e:
                            print(e)


def replay_main(args):
    files_dir = Path(args.replay)
    preview_data = (files_dir / 'vis_data.txt').open(mode='r')
    pos_data = (files_dir / 'pos_data.txt').open(mode='r')

    img_timestamp = None
    img_first_timestamp = None
    replay_start_timestamp = get_time_ms()

    pos_timestamp = None
    pos_first_timestamp = None

    while(True):
        if not img_timestamp:
            # Read from the file that keeps track of timestamps
            img_str = preview_data.readline()

            # No more imgs, exit
            if not img_str:
                print("Replay file empty, exiting")
                # exit(0)

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
        if get_time_ms() - replay_start_timestamp > \
                img_timestamp - img_first_timestamp:

            img_dec = cv2.imdecode(np.frombuffer(
                img_data_file, dtype=np.int8), flags=cv2.IMREAD_COLOR)
            data_dict["preview"][0] = img_dec
            animate_preview()
            img_timestamp = None

        if not pos_timestamp:
            # We read one line of data
            pos_str = pos_data.readline()
            # If the file is empty, we exit the program
            if not pos_str:
                print("Replay file empty, exiting")
                exit(0)

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
                                 'pos_z': float(out.group(4))}

        # If the relative time is correct, we publish the data
        if get_time_ms() - replay_start_timestamp > \
                pos_timestamp - pos_first_timestamp:

            data_dict["pos_x"].append(pos_data_dict["pos_x"])
            data_dict["pos_y"].append(pos_data_dict["pos_y"])
            data_dict["pos_z"].append(pos_data_dict["pos_z"])

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

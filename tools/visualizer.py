# Visualizer tool

import threading
import time
import cv2
import socket
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST = "127.0.0.1"  # Host IP
PORT = 65432  # Port
TOPIC_LIST = ["accel_x", "accel_y", "accel_z", "preview"]  # All topics


# Dictionary for all data
data_dict = {topic: [0] for topic in TOPIC_LIST}
DATA_MAX_LEN = 500

KEYS = ["preview", "pos"]

ax_list = dict()


def animate_pos():
    ax_list["pos"].clear()
    ax_list["pos"].set_title("pos")
    ax_list["pos"].scatter(data["accel_x"], data["accel_y"], data["accel_z"])
    ax_list["pos"].figure.canvas.draw()


def animate_preview():
    ax_list["preview"].clear()
    ax_list["preview"].set_title("preview")
    ax_list["preview"].set_axis_off()
    ax_list["preview"].imshow(data_dict["preview"][0])
    ax_list["preview"].figure.canvas.draw()


def plot_main():
    fig = plt.figure()
    ax_list["preview"] = fig.add_subplot(1, 2, 1)
    ax_list["preview"].set_title("preview")
    ax_list["preview"].set_axis_off()

    ax_list["pos"] = fig.add_subplot(1, 2, 2, projection='3d')
    ax_list["pos"].set_title("accel")
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
                        img_dec = cv2.imdecode(np.frombuffer(buf, dtype=np.int8), flags=cv2.IMREAD_COLOR)

                        data_dict["preview"][0] = img_dec
                        animate_preview()


def main():
    socket_thread = threading.Thread(target = socket_main)
    socket_thread.start()

    plot_main()


if __name__ == '__main__':
    main()

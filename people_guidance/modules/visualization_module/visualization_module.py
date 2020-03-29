import pathlib
import cv2
import ctypes
import numpy as np
import socket

from time import sleep

from ..module import Module

HOST = "127.0.0.1"  # Host IP
PORT = 65432  # Port
PREVIEW_FRAMESIZE = (640, 480)
POS_PLOT_HZ = 20  # Discard before plotting
PREVIEW_PLOT_HZ = 20  # Discard before plotting


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module", outputs=[],
                                                  input_topics=["drivers_module:preview", "drivers_module:accelerations_vis"], log_dir=log_dir)

    def start(self):
        self.logger.info("Starting visualization module...")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))

        pos_last_ms = self.get_time_ms()
        preview_last_ms = self.get_time_ms()
        preview_cnt = 0

        while True:
            # POS DATA HANDLING
            pos_vis = self.get("drivers_module:accelerations_vis")
            if pos_vis and self.get_time_ms() > pos_last_ms + 1000/POS_PLOT_HZ:
                pos_last_ms = self.get_time_ms()

                # Encode position data
                pos_data = np.array([pos_vis["data"]["accel_x"], pos_vis["data"]
                                     ["accel_y"], pos_vis["data"]["accel_z"]], dtype='float32').tobytes()
                # Len of pos_data
                buf_len = np.array([len(pos_data)], dtype='uint32')
                # Encode id to uint8
                buf_id = np.array([1], dtype='uint8')

                # Send ID first
                s.sendall(buf_id)
                # Send the image length beforehand
                s.sendall(buf_len)
                # Send data
                s.sendall(pos_data)

            # PREVIEW IMAGE HANDLING
            preview = self.get("drivers_module:preview")
            if preview and self.get_time_ms() > preview_last_ms + 1000/PREVIEW_PLOT_HZ:
                preview_last_ms = self.get_time_ms()

                # Resize image
                img_rs = self.resize_image(preview["data"])
                # Encode len in uint32
                buf_len = np.array([len(img_rs)], dtype='uint32')
                # Encode id to uint8
                buf_id = np.array([0], dtype='uint8')

                # Send ID first
                s.sendall(buf_id)
                # Send the image length beforehand
                s.sendall(buf_len)
                # Send data
                s.sendall(img_rs)

        # Send EOF to detect end of file
        s.shutdown(socket.SHUT_WR)
        s.close()

    def resize_image(self, img):
        # Decode img to bytes
        img_dec = cv2.imdecode(np.frombuffer(
            img, dtype=np.int8), flags=cv2.IMREAD_COLOR)

        # Resize img
        rs = cv2.resize(img_dec, PREVIEW_FRAMESIZE,
                        interpolation=cv2.INTER_LINEAR)

        # Encode img to jpeg
        img_enc = cv2.imencode('.jpeg', rs)[1].tobytes()

        return img_enc

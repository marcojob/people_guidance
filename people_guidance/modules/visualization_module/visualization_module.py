import pathlib
import cv2
import ctypes
import numpy as np
import socket
import io

from time import sleep
from pathlib import Path

from ..module import Module

HOST = "127.0.0.1"  # Host IP
PORT = 65432  # Port
PREVIEW_FRAMESIZE = (640, 480)
POS_PLOT_HZ = 2  # Discard before plotting
PREVIEW_PLOT_HZ = 20  # Discard before plotting


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module", outputs=[],
                                                  inputs=["drivers_module:preview", "drivers_module:accelerations_vis", "feature_tracking_module:feature_point_pairs_vis"], log_dir=log_dir)
        self.args = args

    def start(self):
        self.logger.info("Starting visualization module...")

        if not self.args.save_visualization:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))

        else:
            self.files_dir = Path(self.args.save_visualization)
            (self.files_dir / 'vis').mkdir(parents=True, exist_ok=True)
            self.preview_counter = 0

            self.preview_data = (
                self.files_dir / 'vis_data.txt').open(mode='w')
            self.pos_data = (self.files_dir / 'pos_data.txt').open(mode='w')

        pos_last_ms = self.get_time_ms()
        preview_last_ms = self.get_time_ms()

        features_dict = dict()

        while True:
            sleep(0.01)
            # POS DATA HANDLING
            pos_vis = self.get("drivers_module:accelerations_vis")
            if pos_vis and self.get_time_ms() > pos_last_ms + 1000/POS_PLOT_HZ:
                pos_last_ms = self.get_time_ms()

                # Encode position data
                pos_buf = np.array([pos_vis["data"]["accel_x"], pos_vis["data"]
                                     ["accel_y"], pos_vis["data"]["accel_z"]], dtype='float32').tobytes()
                # Len of pos_data
                buf_len = np.array([len(pos_buf)], dtype='uint32')
                # Encode id to uint8
                buf_id = np.array([1], dtype='uint8')

                if not self.args.save_visualization:
                    # Send ID first
                    s.sendall(buf_id)
                    # Send the image length beforehand
                    s.sendall(buf_len)
                    # Send data
                    s.sendall(pos_buf)
                else:
                    self.pos_data.write(f"{pos_vis['timestamp']}: " +
                                   f"pos_x: {pos_vis['data']['accel_x']}, " +
                                   f"pos_y: {pos_vis['data']['accel_y']}, " +
                                   f"pos_z: {pos_vis['data']['accel_z']}\n")
                    self.pos_data.flush()

            # PREVIEW IMAGE HANDLING
            preview = self.get("drivers_module:preview")
            if preview and self.get_time_ms() > preview_last_ms + 1000/PREVIEW_PLOT_HZ:
                preview_last_ms = self.get_time_ms()

                # Decode img to bytes
                img_dec = cv2.imdecode(np.frombuffer(
                    preview["data"], dtype=np.int8), flags=cv2.IMREAD_COLOR)

                # Draw matches onto image
                matches = features_dict.get(preview["timestamp"], None)
                img_dec = self.draw_matches(img_dec, matches)

                # Resize image
                img_rs = self.resize_image(img_dec)
                # Encode len in uint32
                buf_len = np.array([len(img_rs)], dtype='uint32')
                # Encode id to uint8
                buf_id = np.array([0], dtype='uint8')

                if not self.args.save_visualization:
                    # Send ID first
                    s.sendall(buf_id)
                    # Send the image length beforehand
                    s.sendall(buf_len)
                    # Send data
                    s.sendall(img_rs)
                else:
                    self.preview_counter += 1
                    # Write image
                    img_name = self.files_dir / 'vis' / f"img_{self.preview_counter:04d}.jpg"
                    img_f = io.open(img_name, 'wb')
                    img_f.write(img_rs)
                    img_f.close()

                    timestamp = preview["timestamp"]
                    self.preview_data.write(f"{self.preview_counter}: {timestamp}\n")
                    self.preview_data.flush()

            features = self.get(
                "feature_tracking_module:feature_point_pairs_vis")
            if features:
                features_dict[features["timestamp"]
                              ] = features["data"]["matches"]

        # Send EOF to detect end of file
        s.shutdown(socket.SHUT_WR)
        s.close()

    def resize_image(self, img):
        # Resize img
        rs = cv2.resize(img, PREVIEW_FRAMESIZE,
                        interpolation=cv2.INTER_LINEAR)

        # Encode img to jpeg
        img_enc = cv2.imencode('.jpeg', rs)[1].tobytes()

        return img_enc

    def draw_matches(self, img, matches):
        RADIUS = 30
        THICKNESS = 3
        if matches is not None:
            print("Found matches")
            for match in matches:
                start_point = (match[0][0], match[0][1])
                end_point = (match[1][0], match[1][0])
                img = cv2.circle(img, start_point, RADIUS,
                                 (255, 0, 0), THICKNESS)
                img = cv2.circle(img, end_point, RADIUS,
                                 (0, 0, 255), THICKNESS)
                img = cv2.arrowedLine(img, start_point, end_point, (0, 255, 0), THICKNESS)
        return img

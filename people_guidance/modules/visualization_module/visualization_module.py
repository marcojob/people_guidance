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
PREVIEW_FRAMESIZE = (64, 48)
POS_PLOT_HZ = 5
REPOINTS_PLOT_HZ = 5
PREVIEW_PLOT_HZ = 20


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module", outputs=[],
                                                  inputs=["drivers_module:preview",
                                                          "position_estimation_module:position_vis",
                                                          "feature_tracking_module:feature_point_pairs_vis",
                                                          "reprojection_module:points3d"],
                                                  log_dir=log_dir)
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

            self.preview_data = (self.files_dir / 'vis_data.txt').open(mode='w')

            self.pos_data = (self.files_dir / 'pos_data.txt').open(mode='w')

            self.repoints_data = (self.files_dir / 'repoints_data.txt').open(mode='w')

        pos_last_ms = None
        vis_pos_last_ms = self.get_time_ms()

        repoints_last_ms = None
        vis_repoints_last_ms = self.get_time_ms()

        preview_last_ms = None
        vis_preview_last_ms = self.get_time_ms()

        features_dict = dict()

        while True:
            #sleep(1.0/PREVIEW_PLOT_HZ)
            # POS DATA HANDLING
            if pos_last_ms is None:
                pos_vis = self.get("position_estimation_module:position_vis")

                pos_last_ms = pos_vis.get("timestamp", None)
                vis_pos_last_ms = self.get_time_ms()
            else:
                pos_vis = self.get("position_estimation_module:position_vis")
                if pos_vis and self.get_time_ms() - vis_pos_last_ms > 1000/POS_PLOT_HZ and pos_vis["timestamp"] - pos_last_ms > 1000/POS_PLOT_HZ:
                    pos_last_ms = pos_vis["timestamp"]
                    vis_pos_last_ms = self.get_time_ms()

                    # Encode position data
                    pos_buf = np.array([pos_vis["data"]["pos_x"],
                                        pos_vis["data"]["pos_y"],
                                        pos_vis["data"]["pos_z"],
                                        pos_vis["data"]["angle_x"],
                                        pos_vis["data"]["angle_y"],
                                        pos_vis["data"]["angle_z"]], dtype='float32').tobytes()
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
                                            f"pos_x: {pos_vis['data']['pos_x']}, " +
                                            f"pos_y: {pos_vis['data']['pos_y']}, " +
                                            f"pos_z: {pos_vis['data']['pos_z']}, " +
                                            f"angle_x: {pos_vis['data']['angle_x']}, " +
                                            f"angle_y: {pos_vis['data']['angle_y']}, " +
                                            f"angle_z: {pos_vis['data']['angle_z']}\n")
                        self.pos_data.flush()

            # REPOINTS HANDLING
            if repoints_last_ms is None:
                repoints = self.get("reprojection_module:points3d")

                repoints_last_ms = repoints.get("timestamp", None)
                vis_repoints_last_ms = self.get_time_ms()
            else:
                repoints = self.get("reprojection_module:points3d")
                if repoints and self.get_time_ms() - vis_repoints_last_ms > 1000/REPOINTS_PLOT_HZ and repoints["timestamp"] - repoints_last_ms > 1000/REPOINTS_PLOT_HZ:
                    repoints_last_ms = repoints["timestamp"]
                    vis_repoints_last_ms = self.get_time_ms()

                    # Encode reprojected points
                    repoints_buf = repoints["data"].astype(dtype='float32').tobytes()
                    # Len of pos_data
                    buf_len = np.array([len(repoints_buf)], dtype='uint32')

                    # Encode id to uint8
                    buf_id = np.array([2], dtype='uint8')

                    if not self.args.save_visualization:
                        # Send ID first
                        s.sendall(buf_id)
                        # Send the image length beforehand
                        s.sendall(buf_len)
                        # Send data
                        s.sendall(repoints_buf)
                    else:
                        timestamp = repoints["timestamp"]
                        for point in repoints["data"][0]:
                            self.repoints_data.write(f"{timestamp}: {point[0]}, {point[1]}, {point[2]}")

                        self.repoints_data.flush()


            # PREVIEW IMAGE HANDLING
            if preview_last_ms is None:
                preview = self.get("drivers_module:preview")

                preview_last_ms = preview.get("timestamp", None)
                vis_preview_last_ms = self.get_time_ms()
            else:
                preview = self.get("drivers_module:preview")
                if preview and self.get_time_ms() - vis_preview_last_ms > 1000/PREVIEW_PLOT_HZ \
                        and preview["timestamp"] - preview_last_ms > 1000/PREVIEW_PLOT_HZ:
                    preview_last_ms = preview["timestamp"]
                    vis_preview_last_ms = self.get_time_ms()

                    # Decode img to bytes
                    img_dec = cv2.imdecode(np.frombuffer(
                        preview["data"]["data"], dtype=np.int8), flags=cv2.IMREAD_COLOR)

                    # Draw matches onto image
                    matches = features_dict.get(preview["timestamp"], None)
                    img_dec = self.draw_matches(img_dec, matches)

                    # Resize image
                    img_rs = self.resize_image(img_dec)
                    # Encode len in uint32
                    buf_len = np.array([len(img_rs)], dtype='uint32')
                    buf_len_b = buf_len.tobytes()
                    # Encode id to uint8
                    buf_id = np.array([0], dtype='uint8')
                    buf_id_b = buf_id.tobytes()

                    if not self.args.save_visualization:
                        # Send ID first
                        s.sendall(buf_id_b)
                        # Send the image length beforehand
                        s.sendall(buf_len_b)
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

                features = self.get("feature_tracking_module:feature_point_pairs_vis")
                if features:
                    features_dict[features["timestamp"]
                                ] = features["data"]["point_pairs"]

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
            shape = matches.shape
            for m in range(shape[2]):
                start_point = (matches[0][0][m], matches[0][1][m])
                end_point = (matches[1][0][m], matches[1][1][m])
                img = cv2.circle(img, start_point, RADIUS,
                                 (255, 0, 0), THICKNESS)
                img = cv2.circle(img, end_point, RADIUS,
                                 (0, 0, 255), THICKNESS)
                img = cv2.arrowedLine(
                    img, start_point, end_point, (0, 255, 0), THICKNESS)
        return img
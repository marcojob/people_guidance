import pathlib
import threading
import cv2
import ctypes
import numpy as np
import io

from time import sleep
from pathlib import Path

from ..module import Module

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial.transform import Rotation as R

POS_PLOT_HZ = 5
REPOINTS_PLOT_HZ = 5
PREVIEW_PLOT_HZ = 20

FIGSIZE = (15,12)
DPI = 100
PLOT_LIM = 50.0

MAX_DATA_LEN = 100

KEYS = ["preview", "pos1", "pos2", "plot_t"]
POS_KEYS = ["pos_x", "pos_y", "pos_z", "angle_x", "angle_y", "angle_z", "3d_pos_x", "3d_pos_y", "3d_pos_z"]

ax_list = dict()
scatter_p1 = None
scatter_p2 = None
plot_t = None
scatter_1 = None
scatter_2 = None
preview_p = None


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module", outputs=[],
                                                  inputs=["feature_tracking_module:feature_point_pairs_vis",
                                                          "reprojection_module:points3d",
                                                          "position_module:position_vis"],
                                                  log_dir=log_dir)
        self.args = args

    def start(self):
        self.logger.info("Starting visualization module...")

        data_thread = threading.Thread(target=self.data_main)
        data_thread.start()

        self.len_points_3d = 0

        self.plot_main()

    def plot_main(self):
        try:
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            ax_list["preview"] = fig.add_subplot(2, 2, 1)
            ax_list["preview"].set_title("preview")
            ax_list["preview"].set_axis_off()

            ax_list["pos1"] = fig.add_subplot(2, 2, 2)
            ax_list["pos1"].set_title("Y-Z plane")

            ax_list["pos2"] = fig.add_subplot(2, 2, 4)
            ax_list["pos2"].set_title("X-Y plane")

            ax_list["plot_t"] = fig.add_subplot(2, 2, 3)

            plt.show()
        except Exception as e:
            print(e)

    def data_main(self):
        self.data_dict = {key: list() for key in POS_KEYS}

        pos_last_ms = None
        vis_pos_last_ms = self.get_time_ms()

        repoints_last_ms = None
        vis_repoints_last_ms = self.get_time_ms()

        preview_last_ms = None
        vis_preview_last_ms = self.get_time_ms()

        features_dict = dict()
        ready_for_plot = True

        while True:
            sleep(1.0/PREVIEW_PLOT_HZ)
            # POS DATA HANDLING
            if pos_last_ms is None:
                pos_vis = self.get("position_module:position_vis")

                pos_last_ms = pos_vis.get("timestamp", None)
                vis_pos_last_ms = self.get_time_ms()
            else:
                pos_vis = self.get("position_module:position_vis")
                if pos_vis and self.get_time_ms() - vis_pos_last_ms > 1000/POS_PLOT_HZ and pos_vis["timestamp"] - pos_last_ms > 1000/POS_PLOT_HZ:
                    pos_last_ms = pos_vis["timestamp"]
                    vis_pos_last_ms = self.get_time_ms()

                    for key in POS_KEYS:
                        if len(self.data_dict[key]) > MAX_DATA_LEN:
                            del self.data_dict[key][0]

                    self.data_dict["pos_x"].append(pos_vis["data"]["x"])
                    self.data_dict["pos_y"].append(pos_vis["data"]["y"])
                    self.data_dict["pos_z"].append(pos_vis["data"]["z"])
                    self.data_dict["angle_x"].append(pos_vis["data"]["roll"])
                    self.data_dict["angle_y"].append(pos_vis["data"]["pitch"])
                    self.data_dict["angle_z"].append(pos_vis["data"]["yaw"])

                    try:
                        self.animate_pos()
                    except Exception as e:
                        self.logger.debug(f"{e}")


            features = self.get("feature_tracking_module:feature_point_pairs_vis")
            if features:
                matches = features["data"]["point_pairs"]
                preview = features["data"]["img"]

                # Draw matches onto image
                self.data_dict["preview"] = self.draw_matches(preview, matches)

                try:
                    self.animate_preview()
                except Exception as e:
                    self.logger.warning(f"{e}")


            points_3d = self.get("reprojection_module:points3d")
            if points_3d:
                self.len_points_3d = points_3d["data"].shape[0]
                self.data_dict["3d_pos_x"] = list()
                self.data_dict["3d_pos_y"] = list()
                self.data_dict["3d_pos_z"] = list()
                for point in points_3d["data"]:
                    # Only consider point in the view of the plot
                    x, y, z = point[0]
                    if not (x < -PLOT_LIM or x > PLOT_LIM or y < -PLOT_LIM or y > PLOT_LIM or z < 0 or z > PLOT_LIM):
                        self.data_dict["3d_pos_x"].append(x)
                        self.data_dict["3d_pos_y"].append(y)
                        self.data_dict["3d_pos_z"].append(z)

                try:
                    self.plot_text_box()
                except Exception as e:
                    self.logger.warning(f"{e}")

                try:
                    self.animate_3d_points()
                except Exception as e:
                    self.logger.warning(f"{e}")

    def plot_text_box(self):
        global plot_t

        text = f' Number of matches: {self.len_points_3d}\n' + \
               f' Number of matches in FoV: {len(self.data_dict["3d_pos_x"])}'

        if plot_t == None:
            # fake plot
            ax_list["plot_t"].plot([0, 10], [0, 10], alpha=0)
            ax_list["plot_t"].set_axis_off()

            # real text
            plot_t = ax_list["plot_t"].text(0, 10, text, fontsize=12)
        else:
            plot_t.set_text(text)
        ax_list["plot_t"].figure.canvas.draw_idle()


    def animate_pos(self):
        global scatter_p1, scatter_p2
        global line_x, line_y, line_z

        # Current position and angles
        pos_x = self.data_dict["pos_x"][-1]
        pos_y = self.data_dict["pos_y"][-1]
        pos_z = self.data_dict["pos_z"][-1]
        angle_x = self.data_dict["angle_x"][-1]
        angle_y = self.data_dict["angle_y"][-1]
        angle_z = self.data_dict["angle_z"][-1]

        r = R.from_rotvec(np.array([0, 0, angle_z])).as_matrix()
        sc_xy = 1
        sc_z = 0.5

        if scatter_p1 == None:
            ax_list["pos1"].set_xlim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos1"].set_ylim((-PLOT_LIM, PLOT_LIM))

            scatter_p1 = ax_list["pos1"].scatter(
                self.data_dict["pos_y"], self.data_dict["pos_z"])

        if scatter_p2 == None:
            ax_list["pos2"].set_xlim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos2"].set_ylim((-PLOT_LIM, PLOT_LIM))

            scatter_p2 = ax_list["pos2"].scatter(
                self.data_dict["pos_x"], self.data_dict["pos_y"])

        else:
            scatter_p1.set_offsets(self.data_dict["pos_y"], self.data_dict["pos_z"])
            scatter_p1.set_offsets(self.data_dict["pos_x"], self.data_dict["pos_y"])

    def animate_preview(self):
        global preview_p
        if preview_p == None:
            ax_list["preview"].set_title("preview")
            ax_list["preview"].set_axis_off()

            preview_p = ax_list["preview"].imshow(self.data_dict["preview"])

            ax_list["preview"].figure.canvas.draw_idle()
        else:
            preview_p.set_data(self.data_dict["preview"])
            ax_list["preview"].figure.canvas.draw_idle()

    def animate_3d_points(self):
        global scatter_1
        if scatter_1 == None:
            ax_list["pos1"].set_xlim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos1"].set_ylim((-PLOT_LIM, PLOT_LIM))

            scatter_1 = ax_list["pos1"].scatter(
                self.data_dict["3d_pos_y"], self.data_dict["3d_pos_z"], c=self.data_dict["3d_pos_x"])

        else:
            data_1 = np.array(self.data_dict["3d_pos_y"])
            data_2 = np.array(self.data_dict["3d_pos_z"])
            data = np.transpose(np.vstack((data_1, data_2)))

            scatter_1.set_offsets(data)
            scatter_1.set_array(np.array(self.data_dict["3d_pos_x"]))

        global scatter_2
        if scatter_2 == None:
            ax_list["pos2"].set_xlim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos2"].set_ylim((-PLOT_LIM, PLOT_LIM))

            scatter_2 = ax_list["pos2"].scatter(
                self.data_dict["3d_pos_x"], self.data_dict["3d_pos_y"], c=self.data_dict["3d_pos_z"])

        else:
            data_1 = np.array(self.data_dict["3d_pos_x"])
            data_2 = np.array(self.data_dict["3d_pos_y"])
            data = np.transpose(np.vstack((data_1, data_2)))

            scatter_2.set_offsets(data)
            scatter_2.set_array(np.array(self.data_dict["3d_pos_z"]))

        ax_list["pos1"].figure.canvas.draw_idle()
        ax_list["pos2"].figure.canvas.draw_idle()

    def draw_matches(self, img, matches):
        RADIUS = 5
        THICKNESS = 3
        if matches is not None:
            shape = matches.shape
            for m in range(shape[2]):
                end_point = (matches[0][0][m], matches[0][1][m])
                start_point = (matches[1][0][m], matches[1][1][m])
                img = cv2.circle(img, start_point, RADIUS,
                                 (255, 0, 0), THICKNESS)
                img = cv2.circle(img, end_point, RADIUS,
                                 (0, 0, 255), THICKNESS)
                img = cv2.arrowedLine(
                    img, start_point, end_point, (0, 255, 0), THICKNESS)
        return img

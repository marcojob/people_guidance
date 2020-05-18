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
PLOT_LIM = 10

MAX_DATA_LEN = 100

KEYS = ["preview", "pos"]
POS_KEYS = ["pos_x", "pos_y", "pos_z", "angle_x", "angle_y", "angle_z", "3d_pos_x", "3d_pos_y", "3d_pos_z"]

ax_list = dict()
scatter_p = None
scatter_r = None
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

        self.plot_main()

    def plot_main(self):
        try:
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            ax_list["preview"] = fig.add_subplot(1, 2, 1)
            ax_list["preview"].set_title("preview")
            ax_list["preview"].set_axis_off()

            ax_list["pos"] = fig.add_subplot(1, 2, 2, projection='3d')
            ax_list["pos"].set_title("pos")
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
            rot_coord = R.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            if points_3d:
                self.data_dict["3d_pos_x"] = list()
                self.data_dict["3d_pos_y"] = list()
                self.data_dict["3d_pos_z"] = list()
                for point in points_3d["data"]:
                    point_r = rot_coord.apply(point[0])
                    self.data_dict["3d_pos_x"].append(point_r[0])
                    self.data_dict["3d_pos_y"].append(point_r[1])
                    self.data_dict["3d_pos_z"].append(point_r[2])

                try:
                    self.animate_3d_points()
                except Exception as e:
                    self.logger.warning(f"{e}")


    def animate_pos(self):
        global scatter_p
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

        if scatter_p == None:
            ax_list["pos"].set_title("pos")
            ax_list["pos"].set_xlim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos"].set_ylim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos"].set_zlim((-0, PLOT_LIM))

            scatter_p = ax_list["pos"].scatter(
                self.data_dict["pos_x"], self.data_dict["pos_y"], self.data_dict["pos_z"], alpha=0.01)

            line_x = ax_list["pos"].plot([pos_x, pos_x + sc_xy*r[0][0]], [pos_y, pos_y + sc_xy*r[0][1]], [pos_z, pos_z + sc_z*r[0][2]])
            line_y = ax_list["pos"].plot([pos_x, pos_x + sc_xy*r[1][0]], [pos_y, pos_y + sc_xy*r[1][1]], [pos_z, pos_z + sc_z*r[1][2]])
            line_z = ax_list["pos"].plot([pos_x, pos_x + sc_xy*r[2][0]], [pos_y, pos_y + sc_xy*r[2][1]], [pos_z, pos_z + sc_z*r[2][2]])

            ax_list["pos"].figure.canvas.draw_idle()
        else:
            scatter_p._offsets3d = (self.data_dict["pos_x"], self.data_dict["pos_y"], self.data_dict["pos_z"])

            line_x[0].set_xdata([pos_x, pos_x + sc_xy*r[0][0]])
            line_x[0].set_ydata([pos_y, pos_y + sc_xy*r[0][1]])
            line_x[0].set_3d_properties([pos_z, pos_z + sc_z*r[0][2]])

            line_y[0].set_xdata([pos_x, pos_x + sc_xy*r[1][0]])
            line_y[0].set_ydata([pos_y, pos_y + sc_xy*r[1][1]])
            line_y[0].set_3d_properties([pos_z, pos_z + sc_z*r[1][2]])

            line_z[0].set_xdata([pos_x, pos_x + sc_xy*r[2][0]])
            line_z[0].set_ydata([pos_y, pos_y + sc_xy*r[2][1]])
            line_z[0].set_3d_properties([pos_z, pos_z + sc_z*r[2][2]])

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
        global scatter_r
        if scatter_r == None:
            ax_list["pos"].set_title("pos")
            ax_list["pos"].set_xlim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos"].set_ylim((-PLOT_LIM, PLOT_LIM))
            ax_list["pos"].set_zlim((-0, PLOT_LIM))

            scatter_r = ax_list["pos"].scatter(
                self.data_dict["3d_pos_x"], self.data_dict["3d_pos_y"], self.data_dict["3d_pos_z"])

            ax_list["pos"].figure.canvas.draw_idle()
        else:
            scatter_r._offsets3d = (self.data_dict["3d_pos_x"], self.data_dict["3d_pos_y"], self.data_dict["3d_pos_z"])
            ax_list["pos"].figure.canvas.draw_idle()

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

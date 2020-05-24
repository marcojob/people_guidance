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
POS_KEYS = ["pos_x", "pos_y", "pos_z", "angle_x", "angle_y", "angle_z", "3d_pos_x", "3d_pos_y", "3d_pos_z", "crit"]

ax_list = dict()
scatter_p1 = None
scatter_p2 = None
plot_t = None
scatter_1 = None
scatter_2 = None
preview_p = None

lock = threading.Lock()


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
        self.last_timestamp = 0
        self.preview_delta_ts = 0

        self.cbar_1 = None
        self.cbar_2 = None
        self.fig = None

        self.save_flag = False

        if self.args.save_visualization:
            save_thread = threading.Thread(target=self.save_main, args=(self.args.save_visualization, ))
            save_thread.start()

        self.plot_main()

    def save_main(self, folder):
        save_path = Path(folder)
        save_cnt = 0
        while True:
            if self.save_flag:
                with lock:
                    save_cnt += 1
                    file = save_path / f"img_{save_cnt:04d}.jpg"
                    plt.savefig(file)
                self.save_flag = False
            else:
                sleep(0.01)

    def plot_main(self):
        try:
            self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            ax_list["preview"] = self.fig.add_subplot(2, 2, 1)
            ax_list["preview"].set_title("preview")
            ax_list["preview"].set_axis_off()

            ax_list["pos1"] = self.fig.add_subplot(2, 2, 2)
            ax_list["pos1"].set_title("Front view point cloud")

            ax_list["pos2"] = self.fig.add_subplot(2, 2, 4)
            ax_list["pos2"].set_title("Top view point cloud")

            ax_list["plot_t"] = self.fig.add_subplot(2, 2, 3)

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
                        #self.animate_pos()
                        pass
                    except Exception as e:
                        self.logger.debug(f"{e}")


            features = self.get("feature_tracking_module:feature_point_pairs_vis")
            if features:
                matches = features["data"]["point_pairs"]
                preview = features["data"]["img"]
                timestamp = features["data"]["timestamp"]

                # Draw matches onto image
                self.data_dict["preview"] = self.draw_matches(preview, matches)

                # Track image timestamps
                self.preview_delta_ts = timestamp - self.last_timestamp
                self.last_timestamp = timestamp

                try:
                    with lock:
                        self.animate_preview()
                except Exception as e:
                    self.logger.warning(f"{e}")
                    print(e)


            points_3d = self.get("reprojection_module:points3d")
            if points_3d:
                self.len_points_3d = points_3d["data"]["cloud"].shape[0]
                self.data_dict["3d_pos_x"] = list()
                self.data_dict["3d_pos_y"] = list()
                self.data_dict["3d_pos_z"] = list()
                self.data_dict["3d_dist"] = list()
                for point in points_3d["data"]["cloud"]:
                    x, y, z = point[0]

                    self.data_dict["3d_pos_x"].append(x)
                    self.data_dict["3d_pos_y"].append(y)
                    self.data_dict["3d_pos_z"].append(z)
                    self.data_dict["3d_dist"].append(np.sqrt(x**2 + y**2 + z**2))

                self.data_dict["crit"].append(points_3d["data"]["crit"])

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
        index = [i for i in range(len(self.data_dict["crit"]))]

        if plot_t == None and len(self.data_dict["crit"]) > 0:
            plot_t = ax_list["plot_t"].scatter(index, self.data_dict["crit"], c=255)
        else:
            ax_list["plot_t"].clear()
            plot_t = ax_list["plot_t"].scatter(index, self.data_dict["crit"])

        ax_list["plot_t"].set_title("Collision likelihood")
        #ax_list["plot_t"].figure.canvas.draw_idle()


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
            scatter_p1 = ax_list["pos1"].scatter(
                self.data_dict["pos_y"], self.data_dict["pos_z"])
        else:
            scatter_p1.set_offsets(self.data_dict["pos_y"], self.data_dict["pos_z"])

        if scatter_p2 == None:
            scatter_p2 = ax_list["pos2"].scatter(
                self.data_dict["pos_x"], self.data_dict["pos_y"])
        else:
            scatter_p2.set_offsets(self.data_dict["pos_x"], self.data_dict["pos_y"])


    def animate_preview(self):
        global preview_p
        if preview_p == None:
            ax_list["preview"].set_title("Camera view")
            ax_list["preview"].set_axis_off()

            preview_p = ax_list["preview"].imshow(self.data_dict["preview"][...,::-1])

            #ax_list["preview"].figure.canvas.draw_idle()
        else:
            preview_p.set_data(self.data_dict["preview"][...,::-1])
            #ax_list["preview"].figure.canvas.draw_idle()

        self.fig.canvas.draw_idle()

        self.save_flag = True

    def animate_3d_points(self):
        global scatter_1, ax_list
        x = self.data_dict["3d_pos_x"]
        y = self.data_dict["3d_pos_y"]
        z = self.data_dict["3d_pos_z"]
        d = self.data_dict["3d_dist"]

        if scatter_1 == None:
            scatter_1 = ax_list["pos1"].scatter(y, z, c=d, vmin=np.min(d), vmax=np.max(d))

            self.cbar_1 = self.fig.colorbar(scatter_1, ax=ax_list["pos1"])

        else:
            data_1 = np.array(y)
            data_2 = np.array(z)
            data = np.transpose(np.vstack((data_1, data_2)))
            scatter_1.set_offsets(data)

            scatter_1.set_array(np.array(d))
            self.cbar_1.mappable.set_clim(np.min(d), np.max(d))

            ax_list["pos1"].ignore_existing_data_limits = True
            ax_list["pos1"].update_datalim(scatter_1.get_datalim(ax_list["pos1"].transData))
            ax_list["pos1"].autoscale_view()

        ax_list["pos1"].invert_xaxis()

        #ax_list["pos1"].figure.canvas.draw_idle()


        global scatter_2
        if scatter_2 == None:
            scatter_2 = ax_list["pos2"].scatter(x, y, c=d, vmin=np.min(d), vmax=np.max(d))

            self.cbar_2 = self.fig.colorbar(scatter_2, ax=ax_list["pos2"])

            ax_list["pos2"].autoscale()

        else:
            data_1 = np.array(x)
            data_2 = np.array(y)
            data = np.transpose(np.vstack((data_1, data_2)))
            scatter_2.set_offsets(data)

            scatter_2.set_array(np.array(d))
            self.cbar_2.mappable.set_clim(np.min(d), np.max(d))

            ax_list["pos2"].ignore_existing_data_limits = True
            ax_list["pos2"].update_datalim(scatter_2.get_datalim(ax_list["pos2"].transData))
            ax_list["pos2"].autoscale_view()

        #ax_list["pos2"].figure.canvas.draw_idle()

    def draw_matches(self, img, matches):
        RADIUS = 5
        THICKNESS = 1
        prev = matches[0]
        cur = matches[1]
        if matches is not None:
            shape = prev.shape
            for m in range(shape[0]):
                end_point = (cur[m][0], cur[m][1])
                start_point = (prev[m][0], prev[m][1])
                img = cv2.arrowedLine(
                    img, start_point, end_point, (0, 255, 0), THICKNESS)
        return img

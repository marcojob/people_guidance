import pathlib
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt

from time import sleep

from ..module import Module

HOST = "localhost" # Host IP
PORT = 1883 # Port


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module", outputs=[],
                                                  input_topics=["drivers_module:preview", "drivers_module:accelerations_vis"]
                                                  , log_dir=log_dir)

        self.display_fps = 0.0

    def start(self):
        self.logger.info("Starting visualization module...")
        client = mqtt.Client()
        client.connect(HOST, PORT, 60)

        while True:
            sleep(0.001)
            imu = self.get("drivers_module:accelerations_vis")
            client.publish("accel_x", imu.get("data", {}).get("accel_x", 0.0))
            client.publish("accel_y", imu.get("data", {}).get("accel_y", 0.0))
            client.publish("accel_z", imu.get("data", {}).get("accel_z", 0.0))

        client.disconnect()


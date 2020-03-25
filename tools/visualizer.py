# Visualizer tool

import threading
import time
import paho.mqtt.client as mqtt

import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST = "localhost" # Host IP
PORT = 1883 # Port
TOPIC_LIST = ["accel_x", "accel_y", "accel_z"] # All topics


# Dictionary for all data
data = {topic: list() for topic in TOPIC_LIST}
DATA_MAX_LEN = 10

KEYS = ["preview", "accel"]

ax_list = dict()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        for topic in TOPIC_LIST:
            client.subscribe(topic)


def on_message(client, userdata, msg):
    if len(data[msg.topic]) > DATA_MAX_LEN:
        del data[msg.topic][0]

    val = float(msg.payload.decode('utf-8'))
    if not val == 0.0:
        data[msg.topic].append(val)
        animate()


# Separate MQTT thread that handles messages
def mqtt_main():
    client = mqtt.Client()
    client.connect(HOST, PORT, 60)

    # Responses
    client.on_connect = on_connect
    client.on_message = on_message

    # Block thread forever
    client.loop_forever()


def animate():
    ax_list["accel"].clear()
    ax_list["accel"].plot([i for i in range(len(data["accel_x"]))], data["accel_x"])
    ax_list["accel"].plot([i for i in range(len(data["accel_y"]))], data["accel_y"])
    ax_list["accel"].plot([i for i in range(len(data["accel_z"]))], data["accel_z"])

    ax_list["accel"].set_title("accel")
    ax_list["accel"].figure.canvas.draw()


def plot_main():
    fig = plt.figure()
    for idx, key in enumerate(KEYS):
        ax_list[key] = fig.add_subplot(1, 2, idx+1)
    plt.show()


def main():
    # Start the MQTT thread that handles communication
    mqtt_thread = threading.Thread(target=mqtt_main)
    mqtt_thread.start()

    plot_main()


if __name__ == '__main__':
    main()

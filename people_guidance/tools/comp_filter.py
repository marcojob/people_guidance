import math
from .simple_dataloader import IMUFrame
from scipy.spatial.transform import Rotation


class ComplementaryFilter:
    def __init__(self):
        self.last_frame = None
        self.tau = 0.5

        self.ang_x = 0.0
        self.ang_y = 0.0
        self.ang_z = 0.0

    def __call__(self, frame: IMUFrame, dt):
        alpha = 0.0004

        self.ang_x += frame.gx * dt
        self.ang_y += frame.gy * dt

        ang_x_acc = math.atan2(frame.ay, frame.az) * 180 / math.pi
        ang_y_acc = math.atan2(frame.ax, frame.az) * 180 / math.pi

        self.ang_x = (1-alpha) * self.ang_y + (alpha * ang_x_acc)
        self.ang_y = (1 - alpha) * self.ang_y + (alpha * ang_y_acc)
        self.ang_z = self.ang_z + frame.gz * dt

        return tuple((self.ang_x, self.ang_y, self.ang_z))


class Integrator:
    def __init__(self, initial_condition=0.):
        self.state = initial_condition

    def __call__(self, value, dt):
        self.state += dt * value
        return self.state


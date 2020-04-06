from typing import List
from collections import Sequence

class Position:
    def __init__(self, ts, x, y, z, roll, pitch, yaw):
        super().__init__()
        self.ts = ts
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __getitem__(self, item):
        return self.__dict__[item]

    def __str__(self):
        return "Position: " + str(self.__dict__)


def new_interpolated_position(timestamp, position0: Position, position1: Position):

    if position1.ts == position0.ts:
        return position1
    else:
        lever = (timestamp - position0.ts) / (position1.ts - position0.ts)

        properties = {"ts": timestamp}
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
            value = position0[key] + ((position1[key] - position0[key]) * lever)
            properties[key] = value
        return Position(**properties)


def new_empty_position():
    return Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

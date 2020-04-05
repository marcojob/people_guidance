
class Position:
    def __init__(self, timestamp, x, y, z, roll, pitch, yaw):
        super().__init__()
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __getitem__(self, item):
        return self.__dict__[item]

    @staticmethod
    def new_interpolate(timestamp, position0, position1):
        assert position0.timestamp <= timestamp <= position1.timestamp, \
            "timestamp for interpolation must lie between the position timestamps."

        if position1.timestamp == position0.timestamp:
            return position1
        else:
            lever = (timestamp - position0.timestamp) / (position1.timestamp - position0.timestamp)

            properties = {"timestamp": timestamp}
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
                value = position0[key] + ((position1[key] - position0[key]) * lever)
                properties[key] = value
            return Position(**properties)

    @staticmethod
    def new_empty():
        return Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

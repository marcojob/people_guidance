import io

from picamera import mmal, mmalobj
from time import sleep, monotonic
from queue import Queue


class FrameHandler():
    def __init__(self):
        self.camera = mmalobj.MMALCamera()  # camera instance
        self.encoder = mmalobj.MMALImageEncoder()  # encoder instance
        self.framesize = (1640, 1232)  # Highest resolution, FPS ratio
        self.q = Queue()  # Queue that holds images

        # Debug variables
        self.t_start = 0
        self.t_stop = 0
        self.ts_now = 0
        self.ts_last = 0
        self.counter = 0

    def start(self):
        # Setup camera pipeline
        self.camera_pipeline_setup()

        # Start camera pipeline
        self.camera_start()
        self.t_start = self.get_time_ms()

        while True:
            if not self.q.empty():
                # Get items from the queue
                data_dict = self.q.get()
                counter = data_dict['counter']
                data = data_dict['data']

                self.ts_last = self.ts_now
                self.ts_now = data_dict['timestamp']

                # Write a new file jpg file for every frame
                f = io.open(f'test/{counter:03d}.jpg', 'wb')
                f.write(data)
                f.close()

                print(f'cnt: {counter:03d}, t: {self.ts_now - self.ts_last}, q: {self.q.qsize()}')

            # Terminate camera after some time
            if self.t_start + 10*1000 < self.get_time_ms():
                self.t_stop = self.get_time_ms()
                self.camera_stop()

                # But still make sure that the queue is emptied beforehand
                if self.q.empty():
                    break
        print(f'avg {1000.0*counter/(self.t_stop - self.t_start)} fps')

    def camera_pipeline_setup(self):
        # Camera output setup
        self.camera.outputs[0].format = mmal.MMAL_ENCODING_RGB24
        self.camera.outputs[0].framesize = self.framesize
        self.camera.outputs[0].framerate = 30
        self.camera.outputs[0].commit()

        # Encoder input setup
        self.encoder.inputs[0].format = mmal.MMAL_ENCODING_RGB24
        self.encoder.inputs[0].framesize = self.framesize
        self.encoder.inputs[0].commit()

        # Encoder output setup
        self.encoder.outputs[0].copy_from(self.encoder.inputs[0])
        self.encoder.outputs[0].format = mmal.MMAL_ENCODING_JPEG
        self.encoder.outputs[0].params[mmal.MMAL_PARAMETER_JPEG_Q_FACTOR] = 90
        self.encoder.outputs[0].commit()

        # Connect encoder input to camera output
        self.encoder.connect(self.camera.outputs[0])
        self.encoder.connection.enable()

    def image_callback(self, port, buf):
        # Is called in separate thread
        self.counter += 1
        # Put data into queue, this is blocking
        # Important to use buf.data instead of context manager, since we want to copy data, not only pointer access
        self.q.put({'data': buf.data, 'counter': self.counter,
                    'timestamp': self.get_time_ms()})
        return False

    def get_time_ms(self):
        # https://www.python.org/dev/peps/pep-0418/#time-monotonic
        return int(round(monotonic() * 1000))

    def camera_start(self):
        self.encoder.outputs[0].enable(self.image_callback)

    def camera_stop(self):
        self.encoder.connection.disable()


if __name__ == '__main__':
    fh = FrameHandler()
    fh.start()

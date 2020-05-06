import pathlib

import pyaudio
import numpy as np

from ..module import Module
from .beeper import Beeper


class UIModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super().__init__(name="ui_module",
                         inputs=["reprojection_module:collision_prob",
                                 "reprojection_module:uncertainty"],
                         log_dir=log_dir)

        self.bitrate = 44100  # bitrate, Hz, must be integer
        self.duration = 2.0
        self.frequency = 800.0
        self.portaudio = None
        self.stream = None


    def start(self):
        phase_offset = 0

        try:
            self.portaudio = pyaudio.PyAudio()
            self.stream = self.portaudio.open(format=pyaudio.paFloat32,
                                              channels=1,
                                              rate=self.bitrate,
                                              output=True)

            while True:
                collision_prob_payload = self.get("reprojection_module:collision_prob")
                if collision_prob_payload:
                    self.duration = (1 - collision_prob_payload["data"])
                    self.logger.critical(f"set duration {self.duration}")


                # generate samples, note conversion to float32 array
                n_samples = int(self.bitrate * self.duration)

                phases = phase_offset + (2 * np.pi * np.arange(n_samples, step=1) * self.frequency / self.bitrate)
                samples = np.sin(phases).astype(np.float32)

                if samples.shape[0] != 0:

                    # this is neccesary to avoid the phase jumping betweens beeps which causes clicking
                    phase_offset = phases[-1]

                    adsr = self.create_adsr_envelope(samples.shape[0], 0.2, 0.0, 0.2)

                    if self.duration > 0.4:
                        self.stream.write(samples * adsr)
                        self.stream.write(np.zeros_like(samples))
                    else:
                        self.stream.write(samples)
        finally:
            if self.portaudio is not None:
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
                if self.portaudio is not None:
                    self.portaudio.terminate()


    @staticmethod
    def create_adsr_envelope(length, attack, decay, release):

        sustain_level = 0.7

        attack_length = int(attack * length)
        release_length = int(release * length)
        sustain_length = length - attack_length - release_length

        attack_ramp = np.linspace(start=0.0, stop=sustain_level, num=attack_length)
        sustain = np.ones(sustain_length) * sustain_level
        release_ramp = np.linspace(start=sustain_level, stop=0.0, num=release_length)

        return np.concatenate((attack_ramp, sustain, release_ramp), axis=0).astype(np.float32)



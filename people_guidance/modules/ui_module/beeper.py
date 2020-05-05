import numpy as np
import multiprocessing as mp

import pyaudio


class Beeper:
    def __init__(self):
        self.duration = mp.Value('d', 0.5)

    def __enter__(self):
        self.audio_handler = mp.Process(target=self.beep, args=(self.duration,))
        self.audio_handler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.audio_handler.terminate()

    def set_duration(self, duration: float):
        self.duration.value = duration

    @staticmethod
    def beep(duration):

        bitrate = 44100  # bitrate, Hz, must be integer
        f = 800.0  # 440.0  # sine duration, Hz, may be float

        portaudio = None
        stream = None

        try:
            portaudio = pyaudio.PyAudio()
            stream = portaudio.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=bitrate,
                                    output=True)

            phase_offset = 0

            while True:
                beep_duration = duration.value
                # generate samples, note conversion to float32 array
                n_samples = int(bitrate * beep_duration)

                phases = phase_offset + (2 * np.pi * np.arange(n_samples, step=1) * f / bitrate)
                samples = (np.sin(phases)).astype(np.float32)

                # this is neccesary to avoid the phase jumping betweens beeps which causes clicking
                phase_offset = phases[-1]

                adsr = adsr_envelope(samples.shape[0], 0.2, 0.0, 0.2)

                if beep_duration > 0.4:
                    stream.write(samples * adsr)
                    stream.write(np.zeros_like(samples))
                else:
                    stream.write(samples)

        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if portaudio is not None:
                portaudio.terminate()


def adsr_envelope(length, attack, decay, release):

    sustain_level = 0.7

    attack_length = int(attack * length)
    release_length = int(release * length)
    sustain_length = length - attack_length - release_length

    attack_ramp = np.linspace(start=0.0, stop=sustain_level, num=attack_length)
    sustain = np.ones(sustain_length) * sustain_level
    release_ramp = np.linspace(start=sustain_level, stop=0.0, num=release_length)

    return np.concatenate((attack_ramp, sustain, release_ramp), axis=0).astype(np.float32)

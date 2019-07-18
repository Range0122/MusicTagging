import math
import logging
import librosa
import numpy as np
import os


class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = pow(2.0, bits_per_sample - 1)
        self.logger = logging.getLogger('balloon_thrift')

    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        # print('cur spl=%f' % self.cur_SPL)
        if is_sil:
            self.logger.debug('cur spl=%f' % self.cur_SPL)
        return is_sil

    def soundPressureLevel(self, chunk):
        value = math.pow(self.localEnergy(chunk), 0.5)
        value = value / len(chunk) + 1e-12
        value = 20.0 * math.log(value, 10)
        return value

    def localEnergy(self, chunk):
        power = 0.0
        for i in range(len(chunk)):
            sample = chunk[i] * self.normal
            power += sample*sample
        return power


def silence(signal, sample_rate, threshold=15):
    """
    threshold: control how much silence to remove
    """
    new_wav = []

    sil_detector = SilenceDetector(threshold)
    for i in range(int(len(signal)/(sample_rate * 0.02))):
        start = int(i * sample_rate * 0.02)
        end = start + int(sample_rate * 0.02)
        is_silence = sil_detector.is_silence(signal[start:end])
        if not is_silence:
            new_wav.extend(signal[start:end])

    return np.array(new_wav), sample_rate

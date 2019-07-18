import wave
import matplotlib.pyplot as plt
import numpy as np


def waveform(path):
    with wave.open(path, 'rb') as f:
        params = f.getparams()
        n_channels, sample_width, frame_rate, n_frames = params[:4]

        pcm_data = f.readframes(n_frames)
        wave_data = np.fromstring(pcm_data, dtype=np.int16)  # 将字符串转化为int
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # wave幅值归一化
        time = np.arange(0, n_frames) * (1.0 / frame_rate)

        plt.plot(time, wave_data)
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform")
        plt.show()


def spectrogram(path):
    with wave.open(path, 'rb') as f:
        params = f.getparams()
        n_channels, sample_width, frame_rate, n_frames = params[:4]

        pcm_data = f.readframes()
        wave_data = np.fromstring(pcm_data, dtype=np.int16)  # 将字符串转化为int
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # wave幅值归一化
        time = np.arange(0, n_frames) * (1.0 / frame_rate)

        plt.specgram(wave_data, Fs=frame_rate, scale_by_freq=True, sides='default')
        plt.xlabel("Time(s)")
        plt.ylabel("Frequency")
        plt.title("Spectrogram")
        plt.show()


if __name__ == "__main__":
    path = '/Users/range/Code/Data/af2019-sr-devset-20190312/data/55debb545a804dcdf6fedcce9626572d.wav'
    waveform(path)

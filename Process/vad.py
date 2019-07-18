import collections
import contextlib
import wave
import os
import numpy as np
import librosa

import webrtcvad


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())

        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    percent_frame = 0.9

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > percent_frame * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > percent_frame * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
    #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')

    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def bytes_reverse(byte_data):
    """
    将bytes型的数据逆序
    输出/出均为bytes型
    """
    int_data = np.fromstring(byte_data, dtype=np.int16)
    int_data = int_data[::-1]
    return bytes(int_data)


def vad(wave_data, sample_rate, mode=1):
    """
    input: wav文件路径
    output: int型信号向量，以及采样率
    mode = {1,2,3}，值越大表示去除的静音部分越多，(mode=0 与 mode=3效果相同)
    vad对bytes型数据进行处理
    """
    # vad处理
    vad = webrtcvad.Vad()
    vad.set_mode(mode)
    frames = frame_generator(30, wave_data, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    temp_wav = b""
    for i, segment in enumerate(segments):
        temp_wav += segment

    # 判断是否过度静音而去除了所有的音频数据
    if temp_wav:
        new_wav = np.frombuffer(temp_wav, dtype=np.int16)
    else:
        new_wav = np.fromstring(wave_data, dtype=np.int16)
        print("Over filtered.")

    return new_wav, sample_rate


def vad_detect(input_wav, mode=1, reverse=False):
    wave_data, sample_rate = read_wave(input_wav)
    signal, sr = vad(wave_data, sample_rate, mode)
    if reverse:
        signal = signal[::-1]
        signal, sr = vad(bytes(signal), sr, mode)
        signal = signal[::-1]
    return signal, sr


from Process.picture import waveform
import matplotlib.pyplot as plt


if __name__ == '__main__':

    path = '/Users/range/Code/Data/af2019-sr-devset-20190312/data/6c6ed1592d3406dfffe9bb2076d3a734.wav'
    waveform(path)

    signal, sr = vad_detect(path, mode=1, reverse=False)
    signal = signal * 1.0 / (max(abs(signal)))
    time = np.arange(0, len(signal)) * (1.0 / sr)

    plt.plot(time, signal)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()

    signal, sr = vad_detect(path, mode=1, reverse=True)
    signal = signal * 1.0 / (max(abs(signal)))  # wave幅值归一化
    time = np.arange(0, len(signal)) * (1.0 / sr)

    plt.plot(time, signal)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()
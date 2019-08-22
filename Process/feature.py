import librosa
import numpy as np
import math
import os


def compute_melgram(audio_path):
    """
    Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    """

    # mel-spectrogram parameters
    # 后面改成适合自己用的
    # SR = 12000
    SR = 22050
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:
        # 太短填充0
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:
        # 太长从末尾开始取
        start = math.floor((n_sample - n_sample_fit) / 2)
        end = math.floor((n_sample + n_sample_fit) / 2)
        # src = src[(n_sample - n_sample_fit) / 2:(n_sample + n_sample_fit) / 2]
        src = src[start:end]

    melgram = librosa.feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS)
    logam = librosa.amplitude_to_db(melgram ** 2, ref=1.0)

    ret = logam[np.newaxis, np.newaxis, :]

    return ret


def generate_data(path):
    """
    path为存放所有的npy文件的目录
    """
    x = []
    y = []
    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']
    for root, dirs, files in os.walk(path):
        file_list = [root + '/' + file for file in files]
        for file in file_list:
            label = file.split('/')[-1].split('.')[0]

            x.append(np.load(file))
            y.append(labels.index(label))

    return x, y


def generate_feature(root_path = '/home/range/Data/GTZAN/data/'):
    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']
    file_path = []

    for label in labels:
        path = root_path + label
        for root, dirs, files in os.walk(path):
            file_path += [root + '/' + file for file in files]

    for path in file_path:
        feature = compute_melgram(path)
        npy_path = '/home/range/Data/MusicFeature/GTZAN/' + path.split('/')[-1]
        np.save(npy_path, feature)


if __name__ == '__main__':
    # path = '/home/range/Data/MusicFeature/GTZAN'
    # generate_data(path)

    generate_feature()

    # test_path = '/home/range/Data/GTZAN/data/blues/blues.00000.au'
    # test_path = '/home/range/Data/GTZAN/data/blues/blues.00001.au'
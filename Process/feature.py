import librosa
import numpy as np
import math
import os
import sys


def compute_fbank(path):
    y, sr = librosa.load(path, sr=None)


def compute_melspectrogram(path):
    y, sr = librosa.load(path, sr=None, duration=29.11)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
    return spectrogram[np.newaxis, :]


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
    SR = 12000
    # SR = 22050
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
        # 太长取最中间长度为n_sample_fit的部分
        start = math.floor((n_sample - n_sample_fit) / 2)
        end = math.floor((n_sample + n_sample_fit) / 2)
        # src = src[(n_sample - n_sample_fit) / 2:(n_sample + n_sample_fit) / 2]
        src = src[start:end]

    melgram = librosa.feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS)
    logam = librosa.amplitude_to_db(melgram ** 2, ref=1.0)

    ret = logam[np.newaxis, :]
    # ret = melgram[np.newaxis, :]

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
            label = labels.index(label)
            feature = np.load(file)

            # x.append(feature[np.newaxis, :])
            x.append(feature)
            y.append(label)
            # from keras.utils import to_categorical
            # y.append(to_categorical(label, 10, 'int'))
            print("x[0]!!!!!!!!!!!!!!!!!!", feature.shape)
    return np.array(x), np.array(y)
    # return x, y


def generate_spectrogram(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate spectrogram from GTZAN dataset
    """

    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']

    data = {'train': [], 'val': [], 'test': []}

    for label in labels:
        path = root_path + label
        for root, dirs, files in os.walk(path):
            file_list = [root + '/' + file for file in files]
            data['train'] += file_list[:70]
            data['val'] += file_list[70:80]
            data['test'] += file_list[80:]

    for item in data.keys():
        i = 0
        for path in data[item]:
            feature = compute_melgram(path)
            # feature = compute_melspectrogram(path)
            npy_path = f"/home/range/Data/MusicFeature/GTZAN/spectrogram/{item}/{path.split('/')[-1][:-3]}"
            np.save(npy_path, feature)

            i += 1
            percent = i/len(data[item])
            progress(percent, width=30)


def generate_raw_waveform(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate raw waveform from GTZAN dataset
    """
    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']

    data = {'train': [], 'val': [], 'test': []}

    for label in labels:
        path = root_path + label
        for root, dirs, files in os.walk(path):
            file_list = [root + '/' + file for file in files]
            data['train'] += file_list[:70]
            data['val'] += file_list[70:80]
            data['test'] += file_list[80:]

    for item in data.keys():
        i = 0
        for path in data[item]:
            signal, sr = librosa.load(path, sr=None)
            feature = signal[np.newaxis, :]
            print(feature.shape)
            exit()
            npy_path = f"/home/range/Data/MusicFeature/GTZAN/raw_waveform/{item}/{path.split('/')[-1][:-3]}"
            np.save(npy_path, feature)

            i += 1
            percent = i/len(data[item])
            progress(percent, width=30)


def progress(percent, width=50):
    if percent > 1:  # 如果百分比大于1的话则取1
        percent = 1
    show_str = ('[%%-%ds]' % width) % (int(percent * width) * '#')
    # 一共50个#，%d 无符号整型数,-代表左对齐，不换行输出，两个% % 代表一个单纯的%，对应的是后面的s，后面为控制#号的个数
    # print(show_str)  #[###############               ] show_str ，每次都输出一次
    print('\r%s %s%%' % (show_str, int(percent * 100)), end='', file=sys.stdout, flush=True)
    # \r 代表调到行首的意思，\n为换行的意思，fiel代表输出到哪，flush=True代表无延迟，立马刷新。第二个%s是百分比


if __name__ == '__main__':
    generate_spectrogram()

    # test_path = '/home/range/Data/GTZAN/data/blues/blues.00001.au'
    # pre = compute_melgram(test_path)
    # now = compute_melspectrogram(test_path)
    # print(pre.shape, now.shape)

    # path = '/home/range/Data/MusicFeature/GTZAN/spectrogram/'
    # x_train, y_train = generate_data(path + 'train')
    # print(x_train[0].shape)
    # x_train = np.matrix(x_train)


import librosa
import numpy as np
import math
import os
import sys
import random


def shuffle_both(a, b):
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(a)
    random.seed(randnum)
    random.shuffle(b)
    return a, b


def compute_mfcc(path):
    y, sr = librosa.load(path, sr=None, duration=29.12)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    accelerate = librosa.feature.delta(delta)

    feature = np.vstack((mfcc, delta, accelerate))

    return feature[np.newaxis, :]


def compute_new_spec(path):
    y, sr = librosa.load(path, sr=None, duration=27.15)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=39)[np.newaxis, :]
    temp = librosa.amplitude_to_db(spectrogram ** 2, ref=1.0)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)[np.newaxis, :]

    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # delta = librosa.feature.delta(mfcc)
    # accelerate = librosa.feature.delta(delta)

    # temp = np.vstack((mfcc, delta, accelerate))[np.newaxis, :]

    # print(temp.shape)
    # print(spectrogram.shape)
    # exit()

    feature = []

    overlap = 0.5
    feature_length = 130
    audio_length = temp.shape[2]
    feature_num = int(math.floor(((audio_length / feature_length) - 1) / (1 - overlap) + 1))

    for i in range(feature_num):
        # print(i * (1 - overlap) * feature_length, (1 + i * (1 - overlap)) * feature_length)
        start = int(i * (1 - overlap) * feature_length)
        end = int((1 + i * (1 - overlap)) * feature_length)
        feature.append(temp[:, :, start: end])
    # spectrogram = spectrogram.reshape(2600,)
    # spectrogram = np.mean(spectrogram, axis=1)

    return feature


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

            x.append(feature)
            # x.append(np.mean(feature, axis=2).flatten())
            # x.append(feature.flatten())
            y.append(label)

    x, y = shuffle_both(x, y)

    return np.array(x), np.array(y)
    # return x, y


def generate_data_svm(path):
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

            # x.append(feature)
            x.append(np.mean(feature, axis=2).flatten())
            # x.append(feature.flatten())
            y.append(label)

    x, y = shuffle_both(x, y)

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
            data['train'] += file_list[:90]
            data['val'] += file_list[90:]
            data['test'] += file_list[90:]

    for item in data.keys():
        i = 0
        for path in data[item]:
            # feature = compute_melgram(path)
            # feature = compute_melspectrogram(path)
            feature = compute_new_spec(path)
            npy_path = f"/home/range/Data/MusicFeature/GTZAN/mfcc/{item}/{path.split('/')[-1][:-3]}"
            np.save(npy_path, feature)

            i += 1
            percent = i/len(data[item])
            progress(percent, width=30)


def generate_short_feature(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate spectrogram from GTZAN dataset
    """

    labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']

    data = {'train': [], 'val': [], 'test': []}

    for label in labels:
        path = root_path + label
        for root, dirs, files in os.walk(path):
            file_list = [root + '/' + file for file in files]
            data['train'] += file_list[:80]
            data['val'] += file_list[80:85]
            data['test'] += file_list[85:]

    for item in data.keys():
        i = 0
        for path in data[item]:
            feature_list = compute_new_spec(path)
            j = 0
            for feature in feature_list:
                npy_path = f"/home/range/Data/MusicFeature/GTZAN/short_spectrogram/{item}/{path.split('/')[-1][:-3]}{j}"
                np.save(npy_path, feature)
                j += 1
                i += 1
                percent = i/(len(data[item])*9)
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
            signal, sr = librosa.load(path, sr=None, duration=29.12)
            feature = signal
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
    # generate_spectrogram()
    # generate_raw_waveform()
    generate_short_feature()

    # test_path = '/home/range/Data/GTZAN/data/blues/blues.00001.au'
    # feature = compute_new_spec(test_path)

    # for item in feature:
    #     print(item.shape)

    # compute_mfcc(test_path)
    # test = compute_new_spec(test_path)
    # pre = compute_melgram(test_path)
    # now = compute_melspectrogram(test_path)
    # print(pre.shape, now.shape)

    # path = '/home/range/Data/MusicFeature/GTZAN/spectrogram/'
    # x_train, y_train = generate_data(path + 'train')
    # print(x_train[0].shape)
    # x_train = np.matrix(x_train)


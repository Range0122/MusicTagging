import librosa
import numpy as np
import math
import os
import sys
import random
from python_speech_features import logfbank, fbank


def shuffle_both(a, b):
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(a)
    random.seed(randnum)
    random.shuffle(b)
    return a, b


def compute_total_feature(path):
    y, sr = librosa.load(path, sr=None, duration=29.12)
    feature = logfbank(y, sr).T

    return feature[np.newaxis, :]


def compute_short_feature(path):
    y, sr = librosa.load(path, sr=None, duration=27.15)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=39)[np.newaxis, :]
    temp = librosa.amplitude_to_db(spectrogram ** 2, ref=1.0)

    feature = []
    overlap = 0.5
    feature_length = 130
    audio_length = temp.shape[2]
    feature_num = int(math.floor(((audio_length / feature_length) - 1) / (1 - overlap) + 1))

    for i in range(feature_num):
        start = int(i * (1 - overlap) * feature_length)
        end = int((1 + i * (1 - overlap)) * feature_length)
        feature.append(temp[:, :, start: end])

    return feature


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


def generate_total_feature(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate feature from GTZAN dataset
    """
    feature_type = 'logfbank'
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
            feature = compute_total_feature(path)
            npy_path = f"/home/range/Data/MusicFeature/GTZAN/{feature_type}/{item}/{path.split('/')[-1][:-3]}"
            np.save(npy_path, feature)

            i += 1
            percent = i/len(data[item])
            progress(percent, width=30)


def generate_short_feature(root_path='/home/range/Data/GTZAN/data/'):
    """
    generate feature from GTZAN dataset
    """
    feature_type = 'logfbank'
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
            feature_list = compute_short_feature(path)
            feature_num = len(feature_list)
            j = 0
            for feature in feature_list:
                npy_path = f"/home/range/Data/MusicFeature/GTZAN/{feature_type}/{item}/{path.split('/')[-1][:-3]}{j}"
                np.save(npy_path, feature)
                j += 1
                i += 1
                percent = i/(len(data[item]) * feature_num)
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
    # generate_short_feature()
    generate_total_feature()

    # test_path = '/home/range/Data/GTZAN/data/blues/blues.00001.au'
    # feature = compute_total_feature(test_path)
    # feature = compute_short_feature(test_path)

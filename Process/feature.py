import librosa
from python_speech_features import fbank, logfbank, mfcc, delta
import scipy.io.wavfile as wav
import numpy as np
import os


def mfcc_feature_extract(wav_file):
    y, sr = librosa.load(wav_file, sr=None)

    mfcc_feature = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_feature = normalize(mfcc_feature)

    norm_feature = librosa.util.normalize(mfcc_feature)

    delta_feature = librosa.feature.delta(norm_feature, order=1)
    accelerate_feature = librosa.feature.delta(norm_feature, order=2)

    feature = np.hstack((norm_feature, delta_feature, accelerate_feature))

    return feature


def normalize(array, length=80):
    while len(array[0]) < length:
        array = np.hstack((array, array))
    while len(array[0]) > length:
        array = np.delete(array, 0, 1)
        if len(array[0]) > length:
            array = np.delete(array, len(array)+1, 1)
    return array


def fbank_feature_extract(wav_file):
    (rate, sig) = wav.read(wav_file)
    feat_mfcc = mfcc(sig, rate)
    feat_fbank = feat_mfcc.T
    feat_fbank = normalize(feat_fbank, length=80)

    feat_fbank_d = delta(feat_fbank, 2)
    feat_fbank_dd = delta(feat_fbank_d, 2)

    wav_feature = np.hstack((feat_fbank, feat_fbank_d, feat_fbank_dd))

    return wav_feature


import librosa
import numpy as np

# path = '/home/range/Data/GTZAN/data/blues/blues.00000.au'
# y, sr = librosa.load(path, sr=None, duration=29.12)
# spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=256)
# logam = librosa.amplitude_to_db(spectrogram ** 2, ref=1.0)

path1 = '/home/range/Data/MusicFeature/GTZAN/spectrogram/train/blues.00019.npy'
feature1 = np.load(path1)

path2 = '/home/range/Data/MusicFeature/GTZAN/mine_spectrogram/train/blues.00019.npy'
feature2 = np.load(path2)

print(feature1.shape, feature2.shape)


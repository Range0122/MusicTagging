from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from Process.feature import generate_data_svm
import numpy as np

path = '/home/range/Data/MusicFeature/GTZAN/stft/'

x_train, y_train = generate_data_svm(path + 'train')
x_val, y_val = generate_data_svm(path + 'val')
x_test, y_test = generate_data_svm(path + 'test')

print('Input Shape: ', x_train[0].shape)
print('X size: ', len(x_train))

print('Start training...')
# print(x_train.shape, y_train.shape)
#
# for item in x_train, x_val, x_test:
#     print(item.shape)
#
# for item in y_train, y_val, y_test:svm_spectrogram
#     print(item.shape)

S = StandardScaler()
S.fit(x_train)
x_train = S.transform(x_train)
x_test = S.transform(x_test)

# model = LogisticRegression(C=1)
model = SVC(C=0.5, kernel='poly', degree=3)
# model = GaussianNB()
# model = LinearSVC()

model.fit(x_train, y_train)
# model.predict(x_test)

print('Start Testing...')

score_test = model.score(x_test, y_test)
score_train = model.score(x_train, y_train)

print('Accuracy_Train:%.2f\nAccuracy_Test:%.2f' % (score_train, score_test))

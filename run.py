from models import *
from Process.feature import generate_data, generate_data_from_MTAT, data_generator_for_MTAT, get_data_shape, load_all_data_GTZAN
import os
import config as C
import tensorflow as tf
import argparse
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.metrics import categorical_accuracy
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=['train', 'test'], help='train or test')
    return parser.parse_args()


def main(args):
    # path = '/home/range/Data/MusicFeature/GTZAN/log_spectrogram/'
    # path = '/home/range/Data/MusicFeature/MTAT/Spectrogram'
    path = '/home/range/Data/MusicFeature/MTAT/log_spectrogram'

    input_shape = (96, 1366, 1)
    output_class = 50
    batch_size = C.BATCH_SIZE

    debug = False
    # debug = True
    if debug:
        print(input_shape)
        exit()

    # model = Basic_GRU(input_shape, output_class)
    model = Basic_CNN(input_shape, output_class)
    # model = CRNN(input_shape, output_class)
    # model = TestRes(input_shape, output_class)
    # model = ResCNN(input_shape, output_class)
    model.summary()

    optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True, decay=1e-6)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if args.target == 'train':
        # x_train, y_train = load_all_data_GTZAN('/'.join((path, 'train')))
        # x_val, y_val = load_all_data_GTZAN('/'.join((path, 'val')))

        # x_train, y_train = generate_data_from_MTAT('/'.join((path, 'train')))
        x_val, y_val = generate_data_from_MTAT('/'.join((path, 'val')))
        # print("TRAIN LENGTH: ", len(x_train))
        # print("VAL LENGTH: ", len(x_val))

        model.fit_generator(data_generator_for_MTAT('/'.join((path, 'train'))), epochs=20,
                            steps_per_epoch=18706 // batch_size,
                            validation_data=(x_val, y_val),
                            validation_steps=len(x_val) // batch_size, verbose=1,
                            callbacks=[ModelCheckpoint(f'check_point/{model.name}_best.h5',
                                                       monitor='val_loss', save_best_only=True, mode='min'),
                                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                                         mode='min'),
                                       EarlyStopping(monitor='val_loss', patience=20)])

        # model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val),
        #           callbacks=[ModelCheckpoint(f'check_point/{model.name}_best.h5', monitor='val_loss',
        #                                      save_best_only=True, mode='min'),
        #                      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min'),
        #                      EarlyStopping(monitor='val_loss', patience=20)])
    else:
        # model.load_weights(f'check_point/{model.name}_best.h5')
        model.load_weights(f'check_point/961366spectrogram/{model.name}_best.h5')

        x_test, y_test = generate_data_from_MTAT('/'.join((path, 'test')))
        # x_test, y_test = load_all_data_GTZAN('/'.join((path, 'test')))

        # print("TEST LENGTH: ", len(x_test))
        # acc = model.evaluate(x_test, y_test, verbose=0)[1]
        # print(acc)

        y_pre = model.predict(x_test)

        rocauc = metrics.roc_auc_score(y_test, y_pre)
        prauc = metrics.average_precision_score(y_test, y_pre, average='macro')
        y_pred = (y_pre > 0.5).astype(np.float32)
        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='samples')

        print(f'Test scores: rocauc={rocauc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')

        # # GTZAN RO-AUC PR-AUC
        # score = model.evaluate(x_test, y_test, verbose=0)
        #
        # print('\n*********Outline*********')
        # print('Test loss:\t%.4f' % score[0])
        # print('Test accuracy:\t%.4f' % score[1])
        #
        # labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']
        # cm = confusion_matrix(y_test, y_pre)
        # print('\n*********Confusion Matrix*********\n', cm)
        # print('\n*********Accuracy Details*********')
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # accuracy = cm.diagonal()
        # for i in range(len(labels)):
        #     print('      {0:<10s}     {1:>.4f}'.format(labels[i], accuracy[i]))
        #
        # result = classification_report(y_test, y_pre, target_names=labels)
        # print('\n*********Classification Report*********\n', result)


if __name__ == "__main__":
    args = get_arguments()
    main(args)

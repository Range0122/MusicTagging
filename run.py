from models import *
from Process.feature import generate_data
import os
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.utils import to_categorical


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=['train', 'test'], help='train or test')
    return parser.parse_args()


def main(args):
    path = '/home/range/Data/MusicFeature/GTZAN/short_mfcc/'

    x_train, y_train = generate_data(path + 'train')
    x_val, y_val = generate_data(path + 'val')
    x_test, y_test = generate_data(path + 'test')

    input_shape = x_train[0].shape
    output_class = 10

    debug = False
    # debug = True
    if debug:
        print(input_shape)
        exit()

    # model = Basic_GRU(input_shape, output_class)
    # model = Basic_CNN(input_shape, output_class)
    model = ResCNN(input_shape, output_class)
    model.summary()

    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if args.target == 'train':
        history = model.fit(x_train, y_train, batch_size=64, epochs=150, validation_data=(x_val, y_val), verbose=1,
                            callbacks=[ModelCheckpoint(f'check_point/{model.name}_best.h5', monitor='val_loss',
                                                       save_best_only=True, mode='min')]
                            )

        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        # plt.legend()
        # plt.show()
    else:
        model.load_weights(f'check_point/{model.name}_best.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        y_pred = model.predict(x_test).argmax(axis=1)

        labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']

        # cm = confusion_matrix(y_test, y_pred, labels)
        result = classification_report(y_test, y_pred, target_names=labels)
        print(result)


if __name__ == "__main__":
    args = get_arguments()
    main(args)

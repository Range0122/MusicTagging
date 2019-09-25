from models import *
from Process.feature import generate_data, generate_data_from_MTAT, data_generator_for_MTAT
import os
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.metrics import categorical_accuracy
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=['train', 'test'], help='train or test')
    return parser.parse_args()


def main(args):
    # path = '/home/range/Data/MusicFeature/GTZAN/short_logfbank/'
    path = '/home/range/Data/MusicFeature/MTAT/Spectrogram'

    x_val, y_val = generate_data_from_MTAT('/'.join((path, 'val')))

    input_shape = x_val[0].shape
    output_class = 50
    batch_size = 128

    debug = False
    # debug = True
    if debug:
        exit()

    model = Basic_GRU(input_shape, output_class)
    # model = Basic_CNN(input_shape, output_class)
    # model = ResCNN(input_shape, output_class)
    model.summary()
    # sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

    if args.target == 'train':
        model.fit_generator(data_generator_for_MTAT('/'.join((path, 'train'))), epochs=50,
                            steps_per_epoch=18706 // batch_size,
                            validation_data=(x_val, y_val), validation_steps=len(x_val) // batch_size, verbose=1,
                            callbacks=[ModelCheckpoint(f'check_point/{model.name}_best.h5',
                                                       monitor='val_loss', save_best_only=True, mode='min'),
                                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                                         mode='min'),
                                       EarlyStopping(monitor='val_loss', patience=20)])
    else:
        model.load_weights(f'check_point/{model.name}_best.h5')
        
        x_test, y_test = generate_data_from_MTAT('/'.join((path, 'test')))
        score = model.evaluate(x_test, y_test, verbose=0)

        print('\n*********Outline*********')
        print('Test loss:\t%.4f' % score[0])
        print('Test accuracy:\t%.4f' % score[1])

        y_pred = model.predict(x_test).argmax(axis=1)

        # labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']
        labels = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
                  'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
                  'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
                  'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
                  'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
                  'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
                  'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
                  'slow', 'classical', 'guitar']

        cm = confusion_matrix(y_test, y_pred)
        print('\n*********Confusion Matrix*********\n', cm)

        print('\n*********Accuracy Details*********')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy = cm.diagonal()
        for i in range(len(labels)):
            # print('%s\t%-.4f' % ())
            print('      {0:<10s}     {1:>.4f}'.format(labels[i], accuracy[i]))

        result = classification_report(y_test, y_pred, target_names=labels)
        print('\n*********Classification Report*********\n', result)


if __name__ == "__main__":
    args = get_arguments()
    main(args)

from keras import Model
from keras.layers import Input, BatchNormalization, Activation, MaxPool2D
from keras.layers import Conv2D, TimeDistributed, GRU, Dense, Dropout, Flatten, Conv1D, GlobalAvgPool1D
import config as c


def Basic_GRU(input_shape, output_class):
    x_in = Input(input_shape, name='input')
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='conv_relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='conv_relu2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = TimeDistributed(Flatten(), name='timedis1')(x)
    # x = GRU(64, return_sequences=True, name='gru1')(x)
    # x = GRU(128, return_sequences=True, name='gru2')(x)
    x = GRU(256, return_sequences=True, name='gru3')(x)
    x = GRU(512, return_sequences=False, name='gru4')(x)
    x = Dropout(c.DROPOUT, name='gru_drop')(x)

    x = Dense(512, name='fc1')(x)
    x = BatchNormalization(name='fc_norm')(x)
    x = Activation('relu', name='fc_relu')(x)

    x = Dense(output_class, activation='softmax', name='fc2')(x)

    return Model(inputs=[x_in], outputs=[x], name='GRU')


def Basic_CNN(input_shape, output_class):
    x_in = Input(input_shape, name='input')
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='conv_relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='conv_relu2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = TimeDistributed(Flatten(), name='timedis1')(x)

    x = GRU(128, return_sequences=True, name='gru3')(x)
    x = GRU(128, return_sequences=False, name='gru4')(x)

    x = Dense(128, name='fc1')(x)
    x = BatchNormalization(name='fc1_norm')(x)
    x = Activation('relu', name='fc1_relu')(x)

    x = Dense(64, name='fc2')(x)
    x = BatchNormalization(name='fc2_norm')(x)
    x = Activation('relu', name='fc2_relu')(x)

    x = Dense(output_class, activation='softmax', name='fc3')(x)

    return Model(inputs=[x_in], outputs=[x], name='GRU')


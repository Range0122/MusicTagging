from keras import Model
from keras.layers import Input, BatchNormalization, Activation, MaxPool2D, Reshape, ELU, Lambda, ZeroPadding2D
from keras.layers import Conv2D, TimeDistributed, GRU, Dense, Dropout, Flatten, LSTM, Add
from keras.regularizers import l2
import keras.backend as K


def Basic_GRU(input_shape, output_class):
    x_in = Input(input_shape, name='input')
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='conv_relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Dropout(0.2, name='dropout1')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='conv_relu2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = Dropout(0.2, name='dropout2')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu', name='conv_relu3')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    x = Dropout(0.2, name='dropout3')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu', name='conv_relu4')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)
    x = Dropout(0.2, name='dropout4')(x)

    x = TimeDistributed(Flatten(), name='timedis1')(x)
    x = GRU(128, return_sequences=True, name='gru1')(x)
    x = GRU(128, return_sequences=False, name='gru2')(x)
    # x = GRU(128, return_sequences=True, name='gru3')(x)
    # x = GRU(128, return_sequences=False, name='gru4')(x)
    x = Dropout(0.3, name='gru_drop')(x)

    x = Dense(64, name='fc1')(x)
    x = BatchNormalization(name='fc_norm')(x)
    x = Activation('relu', name='fc_relu')(x)

    # x = Dense(output_class, activation='softmax', name='fc2')(x)
    x = Dense(output_class, activation='sigmoid', name='fc2')(x)

    return Model(inputs=[x_in], outputs=[x], name='GRU')


def Basic_CNN(input_shape, output_class):
    x_in = Input(input_shape, name='input')
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='conv_relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Dropout(0.3, name='dropout1')(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='conv_relu2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = Dropout(0.3, name='dropout2')(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu', name='conv_relu3')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    x = Dropout(0.3, name='dropout3')(x)

    # x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv4')(x)
    # x = BatchNormalization(name='bn4')(x)
    # x = Activation('relu', name='conv_relu4')(x)
    # x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)
    # x = Dropout(0.3, name='dropout4')(x)
    #
    # x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv5')(x)
    # x = BatchNormalization(name='bn5')(x)
    # x = Activation('relu', name='conv_relu5')(x)
    # x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool5')(x)
    # x = Dropout(0.3, name='dropout5')(x)

    x = TimeDistributed(Flatten(), name='timedis1')(x)

    x = GRU(512, return_sequences=True, name='gru1')(x)
    x = GRU(512, return_sequences=True, name='gru2')(x)
    # x = GRU(512, return_sequences=True, name='gru3')(x)
    # x = GRU(512, return_sequences=False, name='gru4')(x)
    x = Dropout(0.3, name='dropout_gru')(x)

    x = Dense(128, name='fc1')(x)
    x = BatchNormalization(name='fc1_norm')(x)
    x = Activation('relu', name='fc1_relu')(x)

    # x = Dense(128, name='fc2')(x)
    # x = BatchNormalization(name='fc2_norm')(x)
    # x = Activation('relu', name='fc2_relu')(x)

    # x = Reshape((int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3]),))(x)

    x = Dense(output_class, activation='sigmoid', name='final_fc')(x)

    return Model(inputs=[x_in], outputs=[x], name='CNN')


def res_conv_block(x, filters, strides, name):
    filter1, filter2, filter3 = filters
    shortcut = Conv2D(filter3, (1, 1), strides=strides, use_bias=True, name=f'{name}_scut_conv',
                      kernel_regularizer=l2(0), kernel_initializer='glorot_uniform')(x)
    shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)

    # block a
    x = Conv2D(filter1, (1, 1), strides=strides, use_bias=True, name=f'{name}_conva',
               kernel_regularizer=l2(0), kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(name=f'{name}_bna')(x)
    x = Activation('relu', name=f'{name}_relua')(x)
    # block b
    x = Conv2D(filter2, (3, 3), padding='same', use_bias=True, name=f'{name}_convb',
               kernel_regularizer=l2(0), kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(name=f'{name}_bnb')(x)
    x = Activation('relu', name=f'{name}_relub')(x)
    # block c
    x = Conv2D(filter3, (1, 1), use_bias=True, name=f'{name}_convc',
               kernel_regularizer=l2(0), kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(name=f'{name}_bnc')(x)

    x = Add(name=f'{name}_scut')([shortcut, x])
    x = Activation('relu', name=f'{name}_relu1')(x)
    return x


def ResCNN(input_shape, num_class):
    x_in = Input(input_shape, name='input')

    # x = ZeroPadding2D((2, 37), name='zero-padding')(x_in)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='norm1')(x)
    x = Activation('relu', name='relu1')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='norm2')(x)
    x = Activation('relu', name='relu2')(x)

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool2')(x)

    x = res_conv_block(x, [64, 64, 256], strides=(2, 2), name='block1')
    x = res_conv_block(x, [64, 64, 256], strides=(2, 2), name='block2')
    x = res_conv_block(x, [64, 64, 256], strides=(2, 2), name='block3')

    x = res_conv_block(x, [128, 128, 512], strides=(2, 2), name='block4')
    x = res_conv_block(x, [128, 128, 512], strides=(2, 2), name='block5')
    x = res_conv_block(x, [128, 128, 512], strides=(2, 2), name='block6')

    # 减少维数
    x = Lambda(lambda y: K.mean(y, axis=[1, 2]), name='avgpool')(x)

    # the final two fcs
    x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform', name='final_fc')(x)
    x = BatchNormalization(name='final_norm')(x)
    x = Activation('relu', name='final_relu')(x)

    x = Dropout(0.2, name='final_drop')(x)
    x = Dense(num_class, kernel_initializer='glorot_uniform', name='logit')(x)
    x = Activation('sigmoid', name='pred')(x)

    return Model(inputs=[x_in], outputs=[x], name='ResCNN')

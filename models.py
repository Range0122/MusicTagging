from keras import Model, Sequential
from keras.layers import Input, BatchNormalization, Activation, MaxPool2D, Reshape, ELU, Lambda, ZeroPadding2D
from keras.layers import Conv2D, TimeDistributed, GRU, Dense, Dropout, Flatten, LSTM, Add
from keras.regularizers import l2
from keras import backend as K


def Basic_GRU(input_shape, output_class):
    # set for Basic_GRU
    K.set_image_dim_ordering('th')

    x_in = Input(input_shape, name='input')

    conv_units = 256
    gru_units = 128
    fc_units = 64

    x = Conv2D(conv_units, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='conv_relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Dropout(0.2, name='dropout1')(x)

    x = TimeDistributed(Flatten(), name='timedis1')(x)
    x = GRU(gru_units, return_sequences=True, name='gru1')(x)
    x = GRU(gru_units, return_sequences=True, name='gru2')(x)
    x = GRU(gru_units, return_sequences=True, name='gru3')(x)
    x = GRU(gru_units, return_sequences=False, name='gru4')(x)
    x = Dropout(0.3, name='gru_drop')(x)

    x = Dense(fc_units, name='fc1')(x)
    x = BatchNormalization(name='fc_norm')(x)
    x = Activation('relu', name='fc_relu')(x)

    # x = Dense(output_class, activation='softmax', name='final_fc')(x)
    x = Dense(output_class, activation='sigmoid', name='final_fc')(x)

    return Model(inputs=[x_in], outputs=[x], name='Basic_GRU')


def CRNN(input_shape, output_class):
    K.set_image_dim_ordering('th')

    x_in = Input(input_shape, name='input')

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(x_in)
    x = BatchNormalization(name='bn_0_freq')(x)

    # Conv block 1

    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = ELU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ELU()(x)
    x = MaxPool2D((3, 3), strides=(3, 3), padding='same', name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = ELU()(x)
    x = MaxPool2D((4, 4), strides=(4, 4), padding='same', name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = ELU()(x)
    x = MaxPool2D((4, 4), strides=(4, 4), padding='same', name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    x = TimeDistributed(Flatten(), name='timedis1')(x)
    # x = Reshape((x.shape[0], x.shape[1] * x.shape[2]))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)

    x = Dense(output_class, activation='sigmoid', name='output')(x)

    return Model(inputs=[x_in], outputs=[x], name='CRNN')


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
    # x = Activation('sigmoid', name='pred')(x)
    x = Activation('softmax', name='pred')(x)

    return Model(inputs=[x_in], outputs=[x], name='ResCNN')

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


def Basic_CNN(input_shape, output_class):
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
    # x = TimeDistributed(Flatten(), name='timedis1')(x)
    x = Reshape((int(x.shape[1] * x.shape[2] * x.shape[3]), ))(x)

    x = Dense(output_class, activation='sigmoid', name='output')(x)

    return Model(inputs=[x_in], outputs=[x], name='Basic_CNN')


def CRNN(input_shape, output_class):
    K.set_image_dim_ordering('th')

    x_in = Input(input_shape, name='input')

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(x_in)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
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
    x = Activation('sigmoid', name='pred')(x)
    # x = Activation('softmax', name='pred')(x)

    return Model(inputs=[x_in], outputs=[x], name='ResCNN')


def TestRes(input_shape, output_class):
    K.set_image_dim_ordering('th')

    x_in = Input(input_shape, name='input')

    # x = ZeroPadding2D((2, 37), name='zero-padding')(x_in)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='norm1')(x)
    x = Activation('relu', name='relu1')(x)

    shortcut1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv1')(x)
    shortcut1 = BatchNormalization(name=f'shortcut_norm1')(shortcut1)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='norm2')(x)
    x = Activation('relu', name='relu2')(x)

    shortcut2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv2')(x)
    shortcut2 = BatchNormalization(name=f'shortcut_norm2')(shortcut2)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv3')(x)
    x = BatchNormalization(name='norm3')(x)
    x = Activation('relu', name='relu3')(x)

    shortcut3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv3')(x)
    shortcut3 = BatchNormalization(name=f'shortcut_norm3')(shortcut3)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv4')(x)
    x = BatchNormalization(name='norm4')(x)
    x = Activation('relu', name='relu4')(x)

    shortcut4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv4')(x)
    shortcut4 = BatchNormalization(name=f'shortcut_norm4')(shortcut4)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv5')(x)
    x = BatchNormalization(name='norm5')(x)
    x = Activation('relu', name='relu5')(x)

    shortcut5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv5')(x)
    shortcut5 = BatchNormalization(name=f'shortcut_norm5')(shortcut5)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv6')(x)
    x = BatchNormalization(name='norm6')(x)
    x = Activation('relu', name='relu6')(x)

    shortcut6 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv6')(x)
    shortcut6 = BatchNormalization(name=f'shortcut_norm6')(shortcut6)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv7')(x)
    x = BatchNormalization(name='norm7')(x)
    x = Activation('relu', name='relu7')(x)

    shortcut7 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv7')(x)
    shortcut7 = BatchNormalization(name=f'shortcut_norm7')(shortcut7)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv8')(x)
    x = BatchNormalization(name='norm8')(x)
    x = Activation('relu', name='relu8')(x)

    shortcut8 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv8')(x)
    shortcut8 = BatchNormalization(name=f'shortcut_norm8')(shortcut8)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv9')(x)
    x = BatchNormalization(name='norm9')(x)
    x = Activation('relu', name='relu9')(x)

    shortcut9 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv9')(x)
    shortcut9 = BatchNormalization(name=f'shortcut_norm9')(shortcut9)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv10')(x)
    x = BatchNormalization(name='norm10')(x)
    x = Activation('relu', name='relu10')(x)

    shortcut10 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv10')(x)
    shortcut10 = BatchNormalization(name=f'shortcut_norm10')(shortcut10)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv11')(x)
    x = BatchNormalization(name='norm11')(x)
    x = Add(name='add_shortcut1')([shortcut1, x])
    x = Activation('relu', name='relu11')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv12')(x)
    x = BatchNormalization(name='norm12')(x)
    x = Add(name='add_shortcut2')([shortcut2, x])
    x = Activation('relu', name='relu12')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv13')(x)
    x = BatchNormalization(name='norm13')(x)
    x = Add(name='add_shortcut3')([shortcut3, x])
    x = Activation('relu', name='relu13')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv14')(x)
    x = BatchNormalization(name='norm14')(x)
    x = Add(name='add_shortcut4')([shortcut4, x])
    x = Activation('relu', name='relu14')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv15')(x)
    x = BatchNormalization(name='norm15')(x)
    x = Add(name='add_shortcut5')([shortcut5, x])
    x = Activation('relu', name='relu15')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv16')(x)
    x = BatchNormalization(name='norm16')(x)
    x = Add(name='add_shortcut6')([shortcut6, x])
    x = Activation('relu', name='relu16')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv17')(x)
    x = BatchNormalization(name='norm17')(x)
    x = Add(name='add_shortcut7')([shortcut7, x])
    x = Activation('relu', name='relu17')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv18')(x)
    x = BatchNormalization(name='norm18')(x)
    x = Add(name='add_shortcut8')([shortcut8, x])
    x = Activation('relu', name='relu18')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv19')(x)
    x = BatchNormalization(name='norm19')(x)
    x = Add(name='add_shortcut9')([shortcut9, x])
    x = Activation('relu', name='relu19')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv20')(x)
    x = BatchNormalization(name='norm20')(x)
    x = Add(name='add_shortcut10')([shortcut10, x])
    x = Activation('relu', name='relu20')(x)

    # 减少维数
    x = Lambda(lambda y: K.mean(y, axis=[1, 2]), name='avgpool')(x)

    # # the final two fcs
    # x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform', name='final_fc')(x)
    # x = BatchNormalization(name='final_norm')(x)
    # x = Activation('relu', name='final_relu')(x)

    x = Dropout(0.2, name='final_drop')(x)
    x = Dense(output_class, kernel_initializer='glorot_uniform', name='logit')(x)
    x = Activation('sigmoid', name='pred')(x)
    # x = Activation('softmax', name='pred')(x)

    return Model(inputs=[x_in], outputs=[x], name='TestRes')

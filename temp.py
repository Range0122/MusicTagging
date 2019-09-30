from keras import Model, Sequential
from keras.layers import Input, BatchNormalization, Activation, MaxPool2D, Reshape, ELU, Lambda, ZeroPadding2D
from keras.layers import Conv2D, TimeDistributed, GRU, Dense, Dropout, Flatten, LSTM, Add
from keras.regularizers import l2
from keras import backend as K


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
    shortcut6 = BatchNormalization(name=f'shortcut_norm')(shortcut6)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv7')(x)
    x = BatchNormalization(name='norm7')(x)
    x = Activation('relu', name='relu7')(x)

    shortcut7 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv7')(x)
    shortcut7 = BatchNormalization(name=f'shortcut_norm')(shortcut7)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv8')(x)
    x = BatchNormalization(name='norm8')(x)
    x = Activation('relu', name='relu8')(x)

    shortcut8 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv8')(x)
    shortcut8 = BatchNormalization(name=f'shortcut_norm')(shortcut8)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv9')(x)
    x = BatchNormalization(name='norm9')(x)
    x = Activation('relu', name='relu9')(x)

    shortcut9 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv9')(x)
    shortcut9 = BatchNormalization(name=f'shortcut_norm')(shortcut9)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv10')(x)
    x = BatchNormalization(name='norm10')(x)
    x = Activation('relu', name='relu10')(x)

    shortcut10 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='shortcut_conv10')(x)
    shortcut10 = BatchNormalization(name=f'shortcut_norm')(shortcut10)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv11')(x)
    x = BatchNormalization(name='norm11')(x)
    x = Add(name='add_shortcut')([shortcut1, x])
    x = Activation('relu', name='relu11')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv12')(x)
    x = BatchNormalization(name='norm12')(x)
    x = Add(name='add_shortcut')([shortcut2, x])
    x = Activation('relu', name='relu12')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv13')(x)
    x = BatchNormalization(name='norm13')(x)
    x = Add(name='add_shortcut')([shortcut3, x])
    x = Activation('relu', name='relu13')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv14')(x)
    x = BatchNormalization(name='norm14')(x)
    x = Add(name='add_shortcut')([shortcut4, x])
    x = Activation('relu', name='relu14')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv15')(x)
    x = BatchNormalization(name='norm15')(x)
    x = Add(name='add_shortcut')([shortcut5, x])
    x = Activation('relu', name='relu15')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv16')(x)
    x = BatchNormalization(name='norm16')(x)
    x = Add(name='add_shortcut')([shortcut6, x])
    x = Activation('relu', name='relu16')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv17')(x)
    x = BatchNormalization(name='norm17')(x)
    x = Add(name='add_shortcut')([shortcut7, x])
    x = Activation('relu', name='relu17')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv18')(x)
    x = BatchNormalization(name='norm18')(x)
    x = Add(name='add_shortcut')([shortcut8, x])
    x = Activation('relu', name='relu18')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv19')(x)
    x = BatchNormalization(name='norm19')(x)
    x = Add(name='add_shortcut')([shortcut9, x])
    x = Activation('relu', name='relu19')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv20')(x)
    x = BatchNormalization(name='norm20')(x)
    x = Add(name='add_shortcut')([shortcut10, x])
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


if __name__ == "__main__":
    input_shape = (96, 1366, 1)
    output_class = 50
    model = TestRes(input_shape, output_class)
    model.summary()

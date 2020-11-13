import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Dense, UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Add, PReLU, LeakyReLU
from keras.layers.core import Flatten
from keras.models import Model


class Generator(object):
    def __init__(self, shape):
        self.shape = shape

    def get_generator_model(self):
        gfils = 64
        n_residual_blocks = 16

        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding="same")(layer_input)
            d = BatchNormalization(momentum=0.5)(d)
            d = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding="same")(d)
            d = BatchNormalization(momentum=0.5)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            u = Conv2D(256, kernel_size=3, strides=1, padding="same")(layer_input)
            u = UpSampling2D(size=2)(u)
            u = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(u)
            return u

        gen_input = Input(shape=self.shape)
        model = Conv2D(64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)

        res_model = model

        for _ in range(n_residual_blocks):
            model = residual_block(model, gfils)

        model = Conv2D(64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = Add()([res_model, model])

        model = deconv2d(model)
        model = deconv2d(model)

        model = Conv2D(3, kernel_size=9, strides=1, padding="same", activation="tanh")(model)

        return Model(gen_input, model)


class Discriminator(object):

    def __init__(self, shape):
        self.shape = shape

    def get_discriminator_model(self):
        dfils = 64

        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding="same")(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.5)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        dis_input = Input(shape=self.shape)

        model = d_block(dis_input, dfils, 1, False)
        model = d_block(model, dfils, strides=2)
        model = d_block(model, dfils * 2, 1)
        model = d_block(model, dfils * 2, strides=2)
        model = d_block(model, dfils * 4, 1)
        model = d_block(model, dfils * 4, strides=2)
        model = d_block(model, dfils * 8, 1)
        model = d_block(model, dfils * 8, strides=2)
        model = Flatten()(model)
        model = Dense(dfils * 16)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dense(1, activation="sigmoid")(model)

        return Model(dis_input, model)

class VGG_LOSS(object):


    def __init__(self, shape):
        self.shape = shape
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        self.loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.loss_model.trainable = False

    def vgg_loss(self, y_true, y_pred):
        return K.mean(K.square(self.loss_model(y_true) - self.loss_model(y_pred)))

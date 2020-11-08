
import imageio
import numpy as np
import datetime
from skimage import transform
import keras.backend as K
from keras.layers import Input, Dense, UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Add, PReLU, LeakyReLU
from keras.layers.core import Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.initializers import Constant
from numpy import array
from glob import glob


if __name__ == '__main__':

    #device_name = tf.test.gpu_device_name()
    # if device_name != '/device:GPU:0' :
    #  raise SystemError('GPU device not found')
    #print(device_name)

    dfils = 64
    gfils = 64
    channels = 3
    lr_height = 128
    lr_width = 128
    lr_shape = (lr_height, lr_width, channels)

    hr_height = lr_height * 4
    hr_width = lr_width * 4
    hr_shape = (hr_height, hr_width, channels)

    optimizer = Adam(0.0001, 0.9)

    # Building pre-trained VGG19 for extracting general image features

    #vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    #for l in vgg.layers:
    #  l.trainable = False
    #outputs = vgg.layers[9].output
    #vgg_model = Model(inputs = vgg.input, outputs = outputs)
    #vgg_model.trainable = False
    #vgg_model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=hr_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False

    def vgg_loss(y_true, y_pred):
        return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

    # Building discriminator

    def d_block(layer_input, filters, strides=1, bn=True):
        d = Conv2D(filters, kernel_size=3, strides=strides, padding="same")(layer_input)
        if bn:
           d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d


    d0 = Input(hr_shape)
    d1 = d_block(d0, dfils, 1, False)
    d2 = d_block(d1, dfils, strides=2)
    d3 = d_block(d2, dfils * 2, 1)
    d4 = d_block(d3, dfils * 2, strides=2)
    d5 = d_block(d4, dfils * 4, 1)
    d6 = d_block(d5, dfils * 4, strides=2)
    d7 = d_block(d6, dfils * 8, 1)
    d8 = d_block(d7, dfils * 8, strides=2)
    d9 = Dense(dfils * 16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    d11 = Flatten()(d10)
    validity = Dense(1, activation="sigmoid")(d11)

    discriminator_model = Model(d0, validity)
    discriminator_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Building generator

    n_blocks = 16


    def residual_block(layer_input, filters):
        d = Conv2D(filters, kernel_size=3, strides=1, padding="same")(layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        d = PReLU(alpha_initializer=Constant(value=0.2))(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding="same")(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d


    def deconv2d(layer_input):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding="same")(u)
        u = Activation('relu')(u)
        return u


    img_lr = Input(lr_shape)
    c1 = Conv2D(64, kernel_size=9, strides=1, padding="same")(img_lr)
    c2 = PReLU(alpha_initializer=Constant(value=0.2))(c1)

    r = residual_block(c2, gfils)
    for _ in range(n_blocks - 1):
        r = residual_block(r, gfils)

    c3 = Conv2D(64, kernel_size=3, strides=1, padding="same")(r)
    c4 = BatchNormalization(momentum=0.8)(c3)
    c5 = Add()([c2, c4])

    u1 = deconv2d(c5)
    u2 = deconv2d(u1)

    out = Conv2D(3, kernel_size=9, strides=1, padding="same", activation="tanh")(u2)

    generator_model = Model(img_lr, out)

    comb_lr = Input(lr_shape)
    #comb_hr = Input(hr_shape)

    fake_hr = generator_model(comb_lr)

    #fake_features = vgg_model(fake_hr)

    discriminator_model.trainable = False

    comb_validity = discriminator_model(fake_hr)

    combined_model = Model(comb_lr, [comb_validity, fake_hr])
    combined_model.compile(loss=["binary_crossentropy", vgg_loss], loss_weights=[1e-3, 1.], optimizer=optimizer)

    # Training
    epochs = 1000

    start_time = datetime.datetime.now()

    path_hr = glob("srgan_train_data/data_train_HR/*")


    img_count = 500
    train_files = path_hr[:img_count]


    batch_size = 1

    #batch_count = int(img_count / batch_size)

    batch_count = 20


    for epoch in range(epochs):

        for _ in range(batch_count):

            # Discriminator

            rand_nums = np.random.randint(0, img_count, size=batch_size)
            batch_path_hr = []
            for indx in rand_nums:
                batch_path_hr.append(path_hr[indx])
            images_hr = []
            images_lr = []
            for path in batch_path_hr:
                img = imageio.imread(path)
                img = transform.resize(img, (hr_height, hr_width))
                images_hr.append(img)
            train_images_hr = array(images_hr)
            train_images_hr = (train_images_hr.astype(np.float32) - 127.5)/127.5
            for img in images_hr:
                img = transform.downscale_local_mean(img, (4, 4, 1))
                images_lr.append(img)
            train_images_lr = array(images_lr)
            train_images_lr = (train_images_lr.astype(np.float32) - 127.5)/127.5
       
            fake_images_hr = generator_model.predict(train_images_lr)

            valid = np.ones(batch_size)
            fake = np.zeros(batch_size)
            discriminator_model.trainable = True
            d_loss_real = discriminator_model.train_on_batch(train_images_hr, valid)
            d_loss_fake = discriminator_model.train_on_batch(fake_images_hr, fake)
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator

            rand_nums = np.random.randint(0, img_count, size=batch_size)
            batch_path_hr = []
            for indx in rand_nums:
                batch_path_hr.append(path_hr[indx])
            images_hr = []
            images_lr = []
            for path in batch_path_hr:
                img = imageio.imread(path)
                img = transform.resize(img, (hr_height, hr_width))
                images_hr.append(img)
            train_images_hr = array(images_hr)
            train_images_hr = (train_images_hr.astype(np.float32) - 127.5)/127.5
            for img in images_hr:
                img = transform.downscale_local_mean(img, (4, 4, 1))
                images_lr.append(img)
            train_images_lr = array(images_lr)
            train_images_lr = (train_images_lr.astype(np.float32) - 127.5)/127.5

            valid = np.ones(batch_size)
            #image_features = vgg_model.predict(train_images_hr)
            discriminator_model.trainable = False
            g_loss = combined_model.train_on_batch(train_images_lr, [valid, train_images_hr])

        elapsed_time = datetime.datetime.now() - start_time
        print("%d time: %s" % (epoch, elapsed_time))



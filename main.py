import imageio
import numpy as np
import datetime
from skimage import transform
from tensorflow.keras.layers import Input, Dense, UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, Add, PReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from glob import glob






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dfils = 64
    gfils = 64
    channels = 3
    lr_height = 256
    lr_width = 256
    lr_shape = (lr_height, lr_width, channels)

    hr_height = lr_height * 4
    hr_width = lr_width * 4
    hr_shape = (hr_height, hr_width, channels)

    optimizer = Adam(0.0001, 0.9)

    # Building pre-trained VGG19 for extracting general image features

    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    vgg.outputs = [vgg.layers[9].output]
    img = Input(shape=hr_shape)
    img_features = vgg(img)

    vgg_model = Model(img, img_features)
    vgg_model.trainable = False
    vgg_model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])


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
    validity = Dense(1, activation="sigmoid")(d10)

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
    comb_hr = Input(hr_shape)

    fake_hr = generator_model(comb_lr)

    fake_features = vgg_model(fake_hr)

    discriminator_model.trainable = False

    comb_validity = discriminator_model(fake_hr)

    combined_model = Model([comb_lr, comb_hr], [comb_validity, fake_features])
    combined_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer=optimizer)

    # Training
    epochs = 800

    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        # Discriminator
        path_hr = glob("srgan_train_data/data_train_HR/*")

        list_hr = []
        list_lr = []

        batch_image_hr = np.random.choice(path_hr, size=1)
        img = imageio.imread(batch_image_hr[0])
        img_hr = transform.resize(img, (hr_height, hr_width))
        img_lr = transform.downscale_local_mean(img_hr, (4, 4, 1))

        list_hr.append(img_hr)
        list_lr.append(img_lr)

        np_hr = np.array(list_hr) / 127.5 - 1.
        np_lr = np.array(list_lr) / 127.5 - 1.

        fake_hr = generator_model.predict(np_lr)
        patch = int(hr_height/16)
        patchGan = (patch, patch, 1)
        valid = np.ones((1,) + patchGan)
        fake = np.zeros((1,) + patchGan)

        d_loss_real = discriminator_model.train_on_batch(np_hr, valid)
        d_loss_fake = discriminator_model.train_on_batch(fake_hr, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generator
        path_hr = glob("srgan_train_data/data_train_HR/*")

        list_hr = []
        list_lr = []

        batch_image_hr = np.random.choice(path_hr, size=1)
        img = imageio.imread(batch_image_hr[0])
        img_hr = transform.resize(img, (hr_height, hr_width))
        img_lr = transform.downscale_local_mean(img_hr, (4, 4, 1))

        list_hr.append(img_hr)
        list_lr.append(img_lr)

        np_hr = np.array(list_hr) / 127.5 - 1.
        np_lr = np.array(list_lr) / 127.5 - 1.

        valid = np.ones((1, ) + patchGan)
        image_features = vgg_model.predict(np_hr)

        g_loss = combined_model.train_on_batch([np_lr, np_hr], [valid, image_features])

        elapsed_time = datetime.datetime.now() - start_time
        print("%d time: %s" % (epoch, elapsed_time))



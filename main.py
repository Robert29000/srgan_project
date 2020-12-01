import numpy as np
import datetime
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from Models import Generator, Discriminator, VGG_LOSS
from DataLoader import DataLoader


if __name__ == '__main__':

    # device_name = tf.test.gpu_device_name()
    #  if device_name != '/device:GPU:0' :
    #   raise SystemError('GPU device not found')
    # print(device_name)

    channels = 3
    lr_height = 128
    lr_width = 128
    lr_shape = (lr_height, lr_width, channels)

    hr_height = lr_height * 4
    hr_width = lr_width * 4
    hr_shape = (hr_height, hr_width, channels)

    optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    loss = VGG_LOSS(hr_shape)

    # Building discriminator

    discriminator_model = Discriminator(hr_shape).get_discriminator_model()

    discriminator_model.compile(loss="binary_crossentropy", optimizer=optimizer)
    # print(discriminator_model.summary())

    # Building generator

    generator_model = Generator(lr_shape).get_generator_model()
    generator_model.compile(loss=loss.vgg_loss, optimizer=optimizer)
    # print(generator_model.summary())

    # Building combined model

    vgg_model = loss.loss_model

    comb_input = Input(shape=lr_shape)
    x = generator_model(comb_input)
    discriminator_model.trainable = False
    comb_val = discriminator_model(x)
    features = vgg_model(x)

    combined_model = Model(inputs=comb_input, outputs=[comb_val, features])
    combined_model.compile(loss=['binary_crossentropy', 'mse'],
                           loss_weights=[1e-3, 1.],
                           optimizer=optimizer)

    # Training
    epochs = 800

    dataLoader = DataLoader("srgan_train_data/data_train_HR/", 500)

    batch_size = 1

    # batch_count = int(img_count / batch_size)

    batch_count = 20

    start_time = datetime.datetime.now()

    np.random.seed(10)

    for epoch in range(epochs):

        for _ in range(batch_count):
            # Discriminator

            train_images_lr, train_images_hr = dataLoader.get_train_images(batch_size, hr_height, hr_width, 4)

            fake_images_hr = generator_model.predict(train_images_lr)

            valid = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.1
            fake = np.random.random_sample(batch_size) * 0.1
            discriminator_model.trainable = True
            d_loss_real = discriminator_model.train_on_batch(train_images_hr, valid)
            d_loss_fake = discriminator_model.train_on_batch(fake_images_hr, fake)
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator

            train_images_lr, train_images_hr = dataLoader.get_train_images(batch_size, hr_height, hr_width, 4)

            valid = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.1
            discriminator_model.trainable = False
            fake_features = vgg_model.predict(train_images_hr)
            g_loss = combined_model.train_on_batch(train_images_lr, [valid, fake_features])

        elapsed_time = datetime.datetime.now() - start_time
        print("%d time: %s" % (epoch, elapsed_time))

    #generator_model.save("drive/My Drive/saved_models/generator_m_v2.h5")



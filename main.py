import numpy as np
import datetime
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from Models import Generator, Discriminator, VGG_LOSS
from DataLoader import DataLoader



if __name__ == '__main__':

    #device_name = tf.test.gpu_device_name()
    # if device_name != '/device:GPU:0' :
    #  raise SystemError('GPU device not found')
    #print(device_name)
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
    print(discriminator_model.summary())

    # Building generator

    generator_model = Generator(lr_shape).get_generator_model();
    generator_model.compile(loss=loss.vgg_loss, optimizer=optimizer)
    print(generator_model.summary())

    # Building combined model

    def get_combined_model(generator, discriminator):
        discriminator.trainable = False
        comb_input = Input(shape=lr_shape)
        x = generator(comb_input)
        comb_output = discriminator(x)
        combined_model = Model(comb_input, [comb_output, x])
        combined_model.compile(loss=["binary_crossentropy", loss.vgg_loss], loss_weights=[1e-3, 1.], optimizer=optimizer)
        return combined_model

    combined_model = get_combined_model(generator_model, discriminator_model)

    # Training
    epochs = 800

    dataLoader = DataLoader("srgan_train_data/data_train_HR/*", 500)

    batch_size = 1

    #batch_count = int(img_count / batch_size)

    batch_count = 10

    start_time = datetime.datetime.now()

    for epoch in range(epochs):

        for _ in range(batch_count):

            # Discriminator

            train_images_lr, train_images_hr = dataLoader.get_train_images(batch_size, 4)
       
            fake_images_hr = generator_model.predict(train_images_lr)

            valid = np.ones(batch_size)
            fake = np.zeros(batch_size)
            discriminator_model.trainable = True
            d_loss_real = discriminator_model.train_on_batch(train_images_hr, valid)
            d_loss_fake = discriminator_model.train_on_batch(fake_images_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator

            train_images_lr, train_images_hr = dataLoader.get_train_images(batch_size, hr_height, hr_width, 4)

            valid = np.ones(batch_size)
            discriminator_model.trainable = False
            g_loss = combined_model.train_on_batch(train_images_lr, [valid, train_images_hr])

        elapsed_time = datetime.datetime.now() - start_time
        print("%d time: %s" % (epoch, elapsed_time))



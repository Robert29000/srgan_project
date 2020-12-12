import numpy as np
import datetime
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from Models import Generator, Discriminator, VGG_LOSS
from DataLoader import DataLoader
import argparse


def train(lr_height, lr_width, data_dir, model_output_dir, epochs, batch_size, batch_count):

    # device_name = tf.test.gpu_device_name()
    #  if device_name != '/device:GPU:0' :
    #   raise SystemError('GPU device not found')
    # print(device_name)
    scale = 4
    channels = 3
    lr_height = lr_height
    lr_width = lr_width
    lr_shape = (lr_height, lr_width, channels)

    hr_height = lr_height * scale
    hr_width = lr_width * scale
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
    epochs = epochs

    loader = DataLoader(data_dir, 500)

    batch_size = batch_size

    # batch_count = int(img_count / batch_size)

    batch_count = batch_count

    start_time = datetime.datetime.now()

    np.random.seed(10)

    for epoch in range(epochs):

        for _ in range(batch_count):
            # Discriminator

            train_images_lr, train_images_hr = loader.get_train_images(batch_size, hr_height, hr_width, scale)

            fake_images_hr = generator_model.predict(train_images_lr)

            valid = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.1
            fake = np.random.random_sample(batch_size) * 0.1
            discriminator_model.trainable = True
            d_loss_real = discriminator_model.train_on_batch(train_images_hr, valid)
            d_loss_fake = discriminator_model.train_on_batch(fake_images_hr, fake)
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator

            train_images_lr, train_images_hr = loader.get_train_images(batch_size, hr_height, hr_width, scale)

            valid = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.1
            discriminator_model.trainable = False
            fake_features = vgg_model.predict(train_images_hr)
            g_loss = combined_model.train_on_batch(train_images_lr, [valid, fake_features])

        elapsed_time = datetime.datetime.now() - start_time
        print("%d time: %s" % (epoch, elapsed_time))

    generator_model.save(model_output_dir + "generator_m_v2.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRGAN training script")
    parser.add_argument('-l', '--lr_height', type=int, default=128, help='Low res image height')
    parser.add_argument('-w', '--lr_width', type=int, default=128, help='Low res image width')
    parser.add_argument('-i', '--input_dir', default='srgan_train_data/data_train_HR/', help='Input image train directory')
    parser.add_argument('-o', '--output_dir', default='', help='Ouput directory for model')
    parser.add_argument('-e', '--epochs', type=int, default=800, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-c', '--batch_count', type=int, default=20, help='Batch count')

    args = parser.parse_args()

    train(lr_height=args.lr_height, lr_width=args.lr_width,
          data_dir=args.input_dir, model_output_dir=args.output_dir,
          epochs=args.epochs, batch_size=args.batch_size, batch_count=args.batch_count)
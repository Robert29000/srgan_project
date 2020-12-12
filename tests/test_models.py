import pytest

from Models import Generator, Discriminator, VGG_LOSS
from keras.applications.vgg19 import VGG19


@pytest.fixture
def shape():
    shape = (64, 64, 3)
    return shape


@pytest.fixture
def generator(shape):
    return Generator(shape)


@pytest.fixture
def discriminator(shape):
    return Discriminator(shape)


@pytest.fixture
def vgg_loss(shape):
    return VGG_LOSS(shape)


def test_gen_constructor(generator, shape):
    assert len(generator.shape) == len(shape)
    assert generator.shape[0] == shape[0]
    assert generator.shape[1] == shape[1]
    assert generator.shape[2] == shape[2]


def test_gen_model(generator, shape):
    model = generator.get_generator_model()
    assert len(model.input.shape) == 4
    assert model.input.shape[1] == shape[0]
    assert model.input.shape[3] == shape[2]
    assert len(model.output.shape) == 4
    assert model.output.shape[1] == shape[0] * 4
    assert model.output.shape[2] == shape[1] * 4
    assert model.output.shape[3] == shape[2]


def test_dis_constructor(discriminator, shape):
    assert len(discriminator.shape) == len(shape)
    assert discriminator.shape[0] == shape[0]
    assert discriminator.shape[1] == shape[1]
    assert discriminator.shape[2] == shape[2]


def test_dis_model(discriminator):
    model = discriminator.get_discriminator_model()
    assert len(model.input.shape) == 4
    assert len(model.output.shape) == 2
    assert model.output.shape[1] == 1


def test_vgg_constructor(vgg_loss, shape):
    assert len(vgg_loss.shape) == len(shape)
    assert vgg_loss.shape[0] == shape[0]
    assert vgg_loss.shape[1] == shape[1]
    assert vgg_loss.shape[2] == shape[2]


def test_vgg_model(vgg_loss, shape):
    model = vgg_loss.loss_model
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=shape)
    assert not model.trainable
    assert len(model.input.shape) == 4
    assert len(model.output.shape) == len(vgg19.get_layer('block5_conv4').output.shape)

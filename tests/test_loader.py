import pytest

from DataLoader import DataLoader
from numpy import array


@pytest.fixture
def loader():
    return DataLoader(data_path="third_party/srgan_train_data/data_train_HR/", data_count=500)


def test_constructor(loader):
    load = loader
    assert load.data_count == 500
    assert len(load.image_paths) == 500


def test_train_data(loader):
    train_lr, train_hr = loader.get_train_images(2, 396, 396, 4)
    assert len(train_lr.shape) == 4
    assert len(train_hr.shape) == 4
    assert train_lr.shape[0] == 2
    assert train_hr.shape[0] == 2
    assert train_lr.shape[1] == 99
    assert train_hr.shape[1] == 396
    assert train_lr.shape[2] == 99
    assert train_hr.shape[2] == 396
    assert train_lr.shape[3] == 3
    assert train_hr.shape[3] == 3


def test_normalize():
    l = [10, 250, 40, 0, 255]
    data = array(l)
    data = DataLoader.normalize(data)
    print(data)
    assert -1 <= data[0] <= 1
    assert -1 <= data[1] <= 1
    assert -1 <= data[2] <= 1
    assert -1 <= data[3] <= 1
    assert -1 <= data[4] <= 1


def test_denormalize():
    l = [-1, 0.664, 1, -0.33, 0.12]
    data = array(l)
    data = DataLoader.denormalize(data)
    assert 0 <= data[0] <= 255
    assert 0 <= data[1] <= 255
    assert 0 <= data[2] <= 255
    assert 0 <= data[3] <= 255
    assert 0 <= data[4] <= 255
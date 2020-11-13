import numpy as np
import imageio
from glob import glob
from skimage import transform
from numpy import array

class DataLoader(object):

    def __init__(self, data_path=None, data_count=0):
        if data_path is not None:
            dir_path = glob(data_path+"*")
            self.data_count = data_count
            self.image_paths = dir_path[:self.data_count]

    def normalize(self, data):
        return (data.astype(np.float32) - 127.5)/127.5

    def denormalize(self, data):
        data = (data + 1) * 127.5
        return data.astype(np.uint8)

    def get_train_images(self, batch_size, hr_height, hr_width, scale_factor):
        rand_nums = np.random.randint(0, self.data_count, size=batch_size)
        batch_paths = []
        for index in rand_nums:
            batch_paths.append(self.image_paths[index])

        images_hr = []
        images_lr = []

        for path in batch_paths:
            img = imageio.imread(path)
            img = transform.resize(img, (hr_height, hr_width))
            images_hr.append(img)
            img = transform.rescale(img, scale_factor)
            images_lr.append(img)

        train_images_hr = array(images_hr)
        train_images_lr = array(images_lr)
        train_images_hr = self.normalize(train_images_hr)
        train_images_lr = self.normalize(train_images_lr)

        return train_images_lr, train_images_hr
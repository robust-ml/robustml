from abc import ABCMeta, abstractmethod
import os
import numpy as np
import PIL.Image

from . import dataset as d

class Provider(metaclass=ABCMeta):
    @abstractmethod
    def provides(self, dataset):
        '''
        Returns whether or not this provider can provide data for the given
        dataset.
        '''
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        '''
        Returns a tuple (x, y) containing the data and label of the given
        index.
        '''
        raise NotImplementedError

class MNIST:
    def __init__(self):
        # XXX relies on tensorflow, remove this dependency. we don't even
        # explicitly list tensorflow as a dependency in our setup.py.
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    def provides(self, dataset):
        return isinstance(dataset, d.MNIST)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        x = mnist.test.images[index]
        y = mnist.test.labels[index]
        return x, y

class ImageNet:
    def __init__(self, path, shape):
        self._path = path
        self._shape = shape

    def provides(self, dataset):
        return isinstance(dataset, d.ImageNet) and self._shape == dataset.shape

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        data_path = os.path.join(self._path, 'val')
        image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
        assert len(image_paths) == 50000
        labels_path = os.path.join(self._path, 'val.txt')
        with open(labels_path) as labels_file:
            labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
            labels = {os.path.basename(i[0]): int(i[1]) for i in labels}
        path = image_paths[index]
        x = self._load_image(path)
        y = labels[os.path.basename(path)]
        return x, y

    # get centered crop of self._size
    def _load_image(self, path):
        h, w, c = self._shape
        aspect = w / h
        image = PIL.Image.open(path)
        image_aspect = image.width / image.height

        if image_aspect > aspect:
            # image is wider than our aspect ratio
            new_height = image.height
            height_off = 0
            new_width = int(aspect * new_height)
            width_off = (image.width - new_width) // 2
        else:
            # image is taller than our aspect ratio
            new_width = image.width
            width_off = 0
            new_height = int(new_width / aspect)
            height_off = (image.height - new_height) // 2

        # box is (left, upper, right, lower)
        image = image.crop((
            width_off,
            height_off,
            width_off+new_width,
            height_off+new_height
        ))

        image = image.resize((w, h))

        arr = np.asarray(image).astype(np.float32) / 255.0
        if arr.ndim == 2:
            # stack greyscale image
            arr = np.repeat(arr[:,:,np.newaxis], repeats=3, axis=2)
        if arr.shape[2] == 4:
            # remove alpha channel
            arr = arr[:,:,:3]
        assert arr.shape == self._shape
        return arr


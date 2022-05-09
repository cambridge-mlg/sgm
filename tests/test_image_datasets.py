from math import floor
from absl.testing import parameterized
import numpy as np

from src.data.image import get_image_dataset


class ImageDatasetTests(parameterized.TestCase):
    """Tests for images datasets."""

    @parameterized.named_parameters(
        {"testcase_name": "MNIST", "dataset_name": "MNIST"},
        {"testcase_name": "FashionMNIST", "dataset_name": "FashionMNIST"},
        {"testcase_name": "KMNIST", "dataset_name": "KMNIST"},
        {"testcase_name": "SVHN", "dataset_name": "SVHN"},
        {"testcase_name": "CIFAR10", "dataset_name": "CIFAR10"},
        {"testcase_name": "CIFAR100", "dataset_name": "CIFAR100"},
    )
    def test_num_outputs(self, dataset_name):
        datasets = get_image_dataset(dataset_name, val_percent=0.)
        self.assertEqual(len(datasets), 2)

        datasets = get_image_dataset(dataset_name, val_percent=0.1)
        self.assertEqual(len(datasets), 3)

    @parameterized.named_parameters(
        {"testcase_name": "0.1", "val_percent": 0.1},
        {"testcase_name": "0.5", "val_percent": 0.5},
    )
    def test_size_of_val_set(self, val_percent):
        train_data, _, val_data = get_image_dataset('MNIST', val_percent=val_percent)

        n_train = len(train_data)
        n_val = len(val_data)
        n_total = n_train + n_val

        self.assertEqual(floor(n_total * val_percent), n_val)


    @parameterized.named_parameters(
        {"testcase_name": "MNIST", "dataset_name": "MNIST", "flatten_img": False, "correct_shape": (28, 28, 1)},
        {"testcase_name": "SVHN", "dataset_name": "SVHN", "flatten_img": False, "correct_shape": (32, 32, 3)},
        {"testcase_name": "MNIST_flat", "dataset_name": "MNIST", "flatten_img": True, "correct_shape": (784,)},
        {"testcase_name": "SVHN_flat", "dataset_name": "SVHN", "flatten_img": True, "correct_shape": (3072,)},
    )
    def test_shapes(self, dataset_name, flatten_img, correct_shape):
        train_data, _, _ = get_image_dataset(dataset_name, flatten_img=flatten_img)

        img_shape = train_data[0][0].shape

        self.assertEqual(img_shape, correct_shape)




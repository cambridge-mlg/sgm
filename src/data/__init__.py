__all__ = [
    'get_image_dataset',
    'train_val_split_sizes',
    'NumpyLoader',
    'METADATA',
]

from src.data.image import get_image_dataset, train_val_split_sizes
from src.data.numpy_loader import NumpyLoader

METADATA = {
    'image_shape': {
        'MNIST': (28, 28, 1),
        'FashionMNIST': (28, 28, 1),
        'KMNIST': (28, 28, 1),
        'SVHN': (32, 32, 3),
        'CIFAR10': (32, 32, 3),
        'CIFAR100': (32, 32, 3),
    },
    'num_train': {
        'MNIST': 60_000,
        'FashionMNIST': 60_000,
        'KMNIST': 60_000,
        'SVHN': 60_000,
        'CIFAR10': 60_000,
        'CIFAR100': 60_000,
    },
    'num_test': {
        'MNIST': 10_000,
        'FashionMNIST': 10_000,
        'KMNIST': 10_000,
        'SVHN': 10_000,
        'CIFAR10': 10_000,
        'CIFAR100': 10_000,
    }
}
"""Datasets used in examples."""

import array
import gzip
import os
from os import path
import struct
import urllib.request
import numpy as np
import pickle
import tarfile

_DATA = "/tmp/jax_example_data/"

def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        print(f"Downloading {url} to {_DATA}")
        urllib.request.urlretrieve(url, out_file)
        print(f"Finished downloading {url} to {_DATA}")

def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))

def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)

def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    filenames = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for filename in filenames:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels

def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels

def fashion_mnist_raw():
    """Download and parse the raw fashion MNIST dataset."""
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )
    
    filenames = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for filename in filenames:
        _download(base_url + filename, "fashion-" + filename)

    train_images = parse_images(path.join(_DATA, "fashion-train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "fashion-train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "fashion-t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "fashion-t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels

def fashion_mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = fashion_mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels

# https://www.cs.toronto.edu/~kriz/cifar.html
def cifar10_raw():
    """Download and parse the raw fashion MNIST dataset."""
    base_url = "https://www.cs.toronto.edu/~kriz/"
    cifar_dir = "cifar-10-batches-py/"

    def unpickle(file, label_key="labels"):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            # Decode UTF-8
            d_decoded = {}
            for key, value in dict.items():
                d_decoded[key.decode("utf8")] = value
            d = d_decoded
        data = d["data"]
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        labels = np.reshape(labels, (len(labels), 1))
        return data, labels
    
    def extract(filename):
        file = tarfile.open(filename)
        file.extract(cifar_dir + "data_batch_1", _DATA)
        file.extract(cifar_dir + "data_batch_2", _DATA)
        file.extract(cifar_dir + "data_batch_3", _DATA)
        file.extract(cifar_dir + "data_batch_4", _DATA)
        file.extract(cifar_dir + "data_batch_5", _DATA)
        file.extract(cifar_dir + "test_batch", _DATA)
        file.close()
    
    filenames = [
        "cifar-10-python.tar.gz",
    ]

    for filename in filenames:
        _download(base_url + filename, filename)

    extract(path.join(_DATA, "cifar-10-python.tar.gz"))

    num_train_samples = 50000
    train_images = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    train_labels = np.empty((num_train_samples, 1), dtype="uint8")
    for i in range(1, 6):
        # Use broadcasting to merge the 5 training sets into one big set
        (
            train_images[(i - 1) * 10000 : i * 10000, :, :, :],
            train_labels[(i - 1) * 10000 : i * 10000],
        ) = unpickle(path.join(_DATA + cifar_dir, "data_batch_" + str(i)))
    # print(train_images.shape)
    # print(train_labels.shape)
    test_images, test_labels = unpickle(path.join(_DATA + cifar_dir, "test_batch"))

    return train_images, train_labels, test_images, test_labels
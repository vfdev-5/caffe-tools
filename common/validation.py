
#
# Methods to
#

# Python
from os.path import exists
import logging

# Numpy
import numpy as np

# Opencv
import cv2

# Caffe
import caffe


def _blackboxed_image_iterator(image, size):
    """
    Yields image with a black box zone inside and box position indices (x, y), also progress in percent
    """
    total = np.ceil(image.shape[1]*1.0/size) * np.ceil(image.shape[0] * 1.0/size)
    progress = 0
    for i in range(0, image.shape[1], size):
        sx = size if i + size < image.shape[1] else i + size - image.shape[1]
        for j in range(0, image.shape[0], size):
            sy = size if j + size < image.shape[0] else j + size - image.shape[0]
            image_copy = image.copy()
            image_copy[j:j+sy, i:i+sx, :] = 0
            yield image_copy, i, j, progress * 1.0 / total
            progress += 1


def net_forward_pass(image, mean_image, model_path, weights_path):
    """
    Compute forward pass using the model from model_path with weights from weights_path
    :param: image is a ndarray with shape (height, width, channels)
    :param: mean_image is a ndarray with shape (height, width, channels)
    :return: net output
    """
    assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
        "Parameter image is a ndarray with shape (height, width, channels)"
    assert isinstance(mean_image, np.ndarray) and len(mean_image.shape) == 3, \
        "Parameter mean_image is a ndarray with shape (height, width, channels)"

    net = caffe.Net(model_path, weights_path, caffe.TEST)
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

    if mean_image is not None:
        if net.blobs['data'].data.shape[1:] != mean_image.shape:
            # resize
            ms = net.blobs['data'].data.shape[2:]   # (B, C, H, W) -> (H, W)
            mean_image = cv2.resize(mean_image, (ms[1], ms[0]))
            # (H, W, C) -> (C, H, W)
            mean_image = mean_image.transpose((2, 0, 1))
        transformer.set_mean('data', mean_image)    # subtract the dataset-mean value in each channel

    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformed_image = transformer.preprocess('data', image)

    ms = net.blobs['data'].data.shape
    net.blobs['data'].reshape(*((1,) + ms[1:]))
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()
    return output


def net_batch_forward_pass(images, mean_image, model_path, weights_path):
    """
    Compute forward pass using the model from model_path with weights from weights_path
    :param: images is a ndarray with shape (batch_size, height, width, channels)
    :param: mean_image is a ndarray with shape (height, width, channels)
    :return: net output
    """
    assert isinstance(images, np.ndarray) and len(images.shape) == 4, \
        "Parameter image is a ndarray with shape (height, width, channels)"
    assert isinstance(mean_image, np.ndarray) and len(mean_image.shape) == 3, \
        "Parameter mean_image is a ndarray with shape (height, width, channels)"

    net = caffe.Net(model_path, weights_path, caffe.TEST)
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

    if mean_image is not None:
        if net.blobs['data'].data.shape[1:] != mean_image.shape:
            # resize
            ms = net.blobs['data'].data.shape[2:]   # (B, C, H, W) -> (H, W)
            mean_image = cv2.resize(mean_image, (ms[1], ms[0]))
            # (H, W, C) -> (C, H, W)
            mean_image = mean_image.transpose((2, 0, 1))
        transformer.set_mean('data', mean_image)    # subtract the dataset-mean value in each channel

    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    ms = net.blobs['data'].data.shape
    k = images.shape[0]
    transformed_images = np.empty((k,) + ms[1:])
    for i in range(images.shape[0]):
        transformed_images[i, ...] = transformer.preprocess('data', images[i, ...])

    net.blobs['data'].reshape(*((k,) + ms[1:]))
    net.blobs['data'].data[...] = transformed_images

    # perform classification
    output = net.forward()
    return output


def compute_distance(r1, r2):
    """
    Method to compute a scalar from two results of forward pass
    :param r1: an ndarray of shape (K, )
    :param r2: an ndarray of shape (K, )
    :return: a scalar value computed from the first key.
    """
    assert r1.shape == r2.shape, "Input arrays should have same shapes: {} != {}".format(r1.shape, r2.shape)

    # Compare first 5 largest probabilities:
    v = r1 - r2
    # max distance value is sqrt(2), min distance value is 0
    return np.sqrt(np.dot(v, v)) / np.sqrt(2)


def net_heatmap(resolution, image, mean_image, model_path, weights_path, results_distance=compute_distance, batch_size=5, verbose=False):
    """
    Compute a 'heatmap' of the network on the image using black box mask method.
    - Forward pass is computed on the origin image and output is stored
    - A black box zeroes a part of the image and forward pass is computed on the modified image
    -- distance is computed between two results
    - Loop on the positions of the black box and compute the distance

    :param: resolution is a variable between 0 and 1 to compute black box size.
            bbox_size = (1-resolution)*(image.size - 1) + 1
    :param: image is a ndarray with shape (height, width, channels)
    :param: mean_image is a ndarray with shape (height, width, channels)

    :param: results_distance is a function to compute a scalar value from two results of net forward passes
    The signature is func(dict, dict) -> real between 0 and 1


    :return:

    """

    res0 = net_forward_pass(image, mean_image, model_path, weights_path)
    assert len(res0) > 0, "Net forward pass output is empty"
    # Take the first key from the net output
    key = res0.keys()[0]

    heatmap = np.zeros(image.shape[:-1])

    image_size = min(image.shape[0], image.shape[1])
    bbox_size = int((1-resolution)*(image_size - 1) + 1)

    images = np.empty((batch_size, ) + image.shape)
    count = 0
    x_vec = []
    y_vec = []

    def _compute():
        results = net_batch_forward_pass(images, mean_image, model_path, weights_path)
        assert len(results) > 0, "Net forward pass output is empty"
        assert key in results, "The first key is not found in both inputs"
        v0 = res0[key]
        v1 = results[key]
        assert v0.shape[0] == 1 and v1.shape[0] == batch_size, "..."
        for i in range(min(v1.shape[0], count)):
            distance = results_distance(v0[0], v1[i, :])
            heatmap[y_vec[i]:y_vec[i]+bbox_size, x_vec[i]:x_vec[i]+bbox_size] = distance

        if verbose:
            cv2.imshow("Heatmap interactive computation ...", (255.0 * heatmap).astype(np.uint8))
            cv2.waitKey(20)

    for image_copy, x, y, progress_percent in _blackboxed_image_iterator(image, bbox_size):
        logging.log(logging.INFO, "Net heatmap computation : [%s / 100]" % progress_percent)

        if count == batch_size:
            _compute()

            images = np.empty((batch_size, ) + image.shape)
            count = 0
            x_vec = []
            y_vec = []

        images[count, ...] = image_copy
        x_vec.append(x)
        y_vec.append(y)
        count += 1

    if count > 0:
        _compute()

    if verbose:
        cv2.destroyAllWindows()
    return res0, heatmap




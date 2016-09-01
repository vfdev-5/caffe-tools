#
# Methods to
#

# Python
import logging
import copy

# Numpy
import numpy as np

# Opencv
import cv2

# Caffe
import caffe


def _blackboxed_image_iterator(image, size):
    """
    Yields image with a black box zone inside and box position indices (x, y), also progress in percent

    Black boxes are inserted on a copy of the original and the covering is without intersections or overlapping as in a tiling procedure.
    For example, in 1D blackboxed output images are :
    - Original image : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    - black box size = 3
    -> [0,0,0,4,5,6,7,8,9,10,11,12,13,14,15]
    -> [1,2,3,0,0,0,7,8,9,10,11,12,13,14,15]
    -> [1,2,3,4,5,6,0,0,0,10,11,12,13,14,15]
    -> [1,2,3,4,5,6,7,8,9, 0, 0, 0,13,14,15]
    -> [1,2,3,4,5,6,7,8,9,10,11,12, 0, 0, 0]
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


def _blackboxed_image_iterator2(image, size):
    """
    Yields image with a black box zone inside and box position indices (x, y), also progress in percent

    Black boxes are inserted on a copy of the original at all pixels locations
    For example, in 1D blackboxed output images are :
    - Original image : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    - black box size = 3
    -> [0,0,0,4,5,6,7,8,9,10,11,12,13,14,15]
    -> [1,0,0,0,5,6,7,8,9,10,11,12,13,14,15]
    -> [1,2,0,0,0,6,7,8,9,10,11,12,13,14,15]
    -> [1,2,3,0,0,0,7,8,9,10,11,12,13,14,15]
    -> [1,2,3,4,0,0,0,8,9,10,11,12,13,14,15]
    -> [1,2,3,4,5,0,0,0,9,10,11,12,13,14,15]
    -> [1,2,3,4,5,6,0,0,0,10,11,12,13,14,15]
    -> ...
    -> [1,2,3,4,5,6,7,8,9,10,11, 0, 0, 0,15]
    -> [1,2,3,4,5,6,7,8,9,10,11,12, 0, 0, 0]
    """
    total = (image.shape[1] - size) * (image.shape[0] - size)
    progress = 0
    for i in range(image.shape[1] - size):
        for j in range(image.shape[0] - size):
            image_copy = image.copy()
            image_copy[j:j+size, i:i+size, :] = 0
            yield image_copy, i, j, progress * 1.0 / total
            progress += 1


def _compute_distance(r1, r2):
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


class NetValidation(object):
    """
    :class: NetValidation is a structure to compute forward passes

    Usage:

        nv = NetValidation('/path/to/caffe/model.prototxt', '/path/to/caffe/weights.caffemodel')
        nv.mean_image = mean_image  # mean_image.shape = (height, width, channels)
        # nv.verbose = True
        output1 = nv.forward_pass(image)  # image.shape = (height, width, channels)
        output2 = nv.batch_forward_pass(images)  # images.shape = (batch_size, height, width, channels)
        # output1, output2 is a dict, e.g. {'prob': [[...], ...]}
        # resolution -> bbox_size = (1-resolution)*(image.size - 1) + 1
        output, heatmap = nv.tiled_heatmap(resolution=0.95, image=image, results_distance=_compute_distance, batch_size=5)

    """
    def __init__(self, model_path, weights_path):
        self.net = caffe.Net(model_path, weights_path, caffe.TEST)
        self._mean_image = None
        self._transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self._transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self._transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        self.verbose = False
        self.batch_size = 5

    @property
    def mean_image(self):
        """ Mean image used to transform input image during forward pass """
        return self._mean_image

    @mean_image.setter
    def mean_image(self, mean_image):
        """
            Set the mean image.
            :param: mean_image is a ndarray with shape (height, width, channels)
        """
        assert isinstance(mean_image, np.ndarray) and len(mean_image.shape) == 3, \
            "Parameter mean_image is a ndarray with shape (height, width, channels)"
        self._mean_image = mean_image
        if self.net.blobs['data'].data.shape[1:] != self._mean_image.shape:
            # resize
            ms = self.net.blobs['data'].data.shape[2:]   # (B, C, H, W) -> (H, W)
            self._mean_image = cv2.resize(self._mean_image, (ms[1], ms[0]))
            # (H, W, C) -> (C, H, W)
            self._mean_image = self._mean_image.transpose((2, 0, 1))
        self._transformer.set_mean('data', self._mean_image)    # subtract the dataset-mean value in each channel

    def forward_pass(self, image):
        """
        Compute forward pass using the model from model_path with weights from weights_path
        :param: image is a ndarray with shape (height, width, channels)
        :param: mean_image is a ndarray with shape (height, width, channels)
        :return: net output
        """
        assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
            "Parameter image is a ndarray with shape (height, width, channels)"

        transformed_image = self._transformer.preprocess('data', image)

        ms = self.net.blobs['data'].data.shape
        self.net.blobs['data'].reshape(*((1,) + ms[1:]))
        self.net.blobs['data'].data[...] = transformed_image

        # perform classification
        output = self.net.forward()
        return copy.deepcopy(output)

    def batch_forward_pass(self, images):
        """
        Compute forward pass using the model from model_path with weights from weights_path
        :param: images is a ndarray with shape (batch_size, height, width, channels)
        :return: net output
        """
        assert isinstance(images, np.ndarray) and len(images.shape) == 4, \
            "Parameter image is a ndarray with shape (height, width, channels)"

        ms = self.net.blobs['data'].data.shape
        k = images.shape[0]
        transformed_images = np.empty((k,) + ms[1:])

        for i in range(images.shape[0]):
            transformed_images[i, ...] = self._transformer.preprocess('data', images[i, ...])

        self.net.blobs['data'].reshape(*((k,) + ms[1:]))
        self.net.blobs['data'].data[...] = transformed_images

        # perform classification
        output = self.net.forward()
        return copy.deepcopy(output)

    def tiled_heatmap(self, resolution, image, results_distance=_compute_distance):
        """
        Compute a 'heatmap' of the network on the image using black box mask method.
        - Forward pass is computed on the origin image and output is stored
        - A black box zeroes a part of the image and forward pass is computed on the modified image
        -- distance is computed between two results
        - Loop on the positions of the black box and compute the distance
        :param: resolution is a variable between 0 and 1 to compute black box size.
                bbox_size = (1-resolution)*(image.size - 1) + 1
        :param: image is a ndarray with shape (height, width, channels)
        :param: results_distance is a function to compute a scalar value from two results of net forward passes
        The signature is func(dict, dict) -> real lied between 0 and 1
        :return: forward pass result dictionary, heatmap
        """

        res0 = self.forward_pass(image)
        assert len(res0) > 0, "Net forward pass output is empty"
        # Take the first key from the net output
        key = res0.keys()[0]

        heatmap = np.ones(image.shape[:-1])

        image_size = min(image.shape[0], image.shape[1])
        bbox_size = int((1-resolution)*(image_size - 1) + 1)

        images = np.zeros((self.batch_size, ) + image.shape)
        count = 0
        x_vec = []
        y_vec = []

        def _compute():
            results = self.batch_forward_pass(images)
            assert len(results) > 0, "Net forward pass output is empty"
            assert key in results, "The first key is not found in both inputs"
            v0 = res0[key]
            v1 = results[key]
            assert v0.shape[0] == 1 and v1.shape[0] == self.batch_size, "..."
            for _x, _y, v in zip(x_vec, y_vec, v1):
                distance = results_distance(v0[0], v)
                heatmap[_y:_y+bbox_size, _x:_x+bbox_size] = distance

            if self.verbose:
                cv2.imshow("Heatmap interactive computation ...", (255.0 * heatmap).astype(np.uint8))
                cv2.waitKey(20)

        for image_copy, x, y, progress_percent in _blackboxed_image_iterator(image, bbox_size):
            logging.log(logging.INFO, "Net heatmap computation : [%s / 100]" % progress_percent)

            if count == self.batch_size:
                _compute()
                images = np.zeros((self.batch_size, ) + image.shape)
                count = 0
                x_vec = []
                y_vec = []

            images[count, ...] = image_copy
            x_vec.append(x)
            y_vec.append(y)
            count += 1

        if count > 0:
            _compute()

        if self.verbose:
            cv2.destroyAllWindows()
        return res0, heatmap

    def full_heatmap(self, bbox_size, image, results_distance=_compute_distance):
        """
        Compute a 'heatmap' of the network on the image using black box mask method.
        - Forward pass is computed on the origin image and output is stored
        - A black box zeroes a part of the image and forward pass is computed on the modified image
        -- distance is computed between two results
        - Loop on the positions of the black box and compute the distance
        :param: bbox_size is the black box size.
        :param: image is a ndarray with shape (height, width, channels)
        :param: results_distance is a function to compute a scalar value from two results of net forward passes
        The signature is func(dict, dict) -> real lied between 0 and 1
        :return: forward pass result dictionary, heatmap
        This method computes 'image.width * image.height' of forward passes, thus this computation can take a lot of time !

        """
        res0 = self.forward_pass(image)
        assert len(res0) > 0, "Net forward pass output is empty"
        # Take the first key from the net output
        key = res0.keys()[0]

        heatmap = 0.01*np.ones(image.shape[:-1])
        images = np.empty((self.batch_size, ) + image.shape)
        count = 0
        x_vec = []
        y_vec = []

        def _compute():
            results = self.batch_forward_pass(images)
            assert len(results) > 0, "Net forward pass output is empty"
            assert key in results, "The first key is not found in both inputs"
            v0 = res0[key]
            v1 = results[key]
            assert v0.shape[0] == 1 and v1.shape[0] == self.batch_size, "..."
            for _x, _y, v in zip(x_vec, y_vec, v1):
                distance = results_distance(v0[0], v)
                heatmap[_y, _x] = distance

            if self.verbose:
                cv2.imshow("Heatmap interactive computation ...", (255.0 * heatmap).astype(np.uint8))
                cv2.waitKey(20)

        for image_copy, x, y, _ in _blackboxed_image_iterator2(image, bbox_size):

            if count == self.batch_size:
                _compute()
                images = np.empty((self.batch_size, ) + image.shape)
                count = 0
                x_vec = []
                y_vec = []

            images[count, ...] = image_copy
            x_vec.append(x + bbox_size/2)
            y_vec.append(y + bbox_size/2)
            count += 1

        if count > 0:
            _compute()

        if self.verbose:
            cv2.destroyAllWindows()
        return res0, heatmap

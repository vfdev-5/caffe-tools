
#
# I/O methods
#

# Opencv
import cv2

# Lmdb
import lmdb
import caffe
from caffe.proto import caffe_pb2


def is_lmdb_file(filename):
    try:
        env = lmdb.open(filename, readonly=True)
        env.close()
    except lmdb.Error:
        return False
    return True


def write_binproto_image(data, filename):
    """
    Write a binproto file with shape (1, channel, height, width)
    :param data: image data with shape, e.g. (height, width, channel)
    :param filename: output filename
    """
    data = data.transpose((2, 0, 1))
    data = data.reshape((1, ) + data.shape)
    blob = caffe.io.array_to_blobproto(data).SerializeToString()
    with open(filename, 'wb') as f:
        f.write(blob)


def read_binproto_image(filename):
    """
    Read binproto image as numpy array. Typical shape is (1, channel, height, width)
    :param filename: input binproto filename
    :return: ndarray
    """
    blob = caffe_pb2.BlobProto()
    with open(filename, 'rb') as f:
        blob.ParseFromString(f.read())
    return caffe.io.blobproto_to_array(blob)


def lmdb_images_iterator(lmdb_file):
    """
    :param lmdb_file:
    :return:
    """
    env = lmdb.open(lmdb_file)
    with env.begin() as lmdb_txn:
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            yield data, label
    env.close()


def read_image(filename, size=None):
    """
    Read image from file
    :param filename: path to image
    :param size: list of width and height, e.g [100, 120]. Can be None
    :return: ndarray with image data. ndarray.shape = (height, widht, number of channels)
    """
    img = cv2.imread(filename) # img.shape = (H, W, Ch)
    if len(img.shape) == 3 and img.shape[2] == 3: # if image has 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = img.reshape(img.shape + (1,))
    if size is not None:
        assert len(size) == 2 and size[0] > 0 and size[1] > 0, "Size should be a list of 2 positive values"
        img = cv2.resize(img, (size[0], size[1]))
    return img

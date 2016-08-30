
"""
    Script to classify a single image

    Usage:
        python single_image_analyze.py <arguments>

"""

# Python
import argparse
import sys
from os.path import join, abspath, dirname, exists

# Matplotlib
import matplotlib.pyplot as plt

# Setup common package path and CAFFE_PATH
COMMON_PACKAGE_PATH = abspath(join(dirname(abspath(__file__)), ".."))
sys.path.insert(0, COMMON_PACKAGE_PATH)
from common.config import cfg
sys.path.insert(0, join(cfg['CAFFE_PATH'], 'python'))

# import helping methods
from common import find_file


#
# Main script
#

parser = argparse.ArgumentParser()
parser.add_argument('path', metavar='test_image_path', type=str,
                    help='Relative to DATASET_PATH or absolute path to the test image')
parser.add_argument('model', metavar='model_file_path', type=str,
                    help='Relative to one of MODELS_PATH_LIST or absolute path to the model prototxt file')
parser.add_argument('weights', metavar='weights_file_path', type=str,
                    help='Relative to one of RESOURCES_PATH_LIST or absolute path to the weights .caffemodel file')
parser.add_argument('--mean', metavar='mean_image_file_path', type=str,
                    help='Relative to one of RESOURCES_PATH_LIST or absolute path to the mean image .binproto file')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

image_path = args.path if exists(args.path) else find_file(args.path, [cfg['DATASET_PATH']])
model_path = args.model if exists(args.model) else find_file(args.model, cfg['MODELS_PATH_LIST'])
weights_path = args.weights if exists(args.weights) else find_file(args.weights, cfg['RESOURCES_PATH_LIST'])
mean_image_path = args.mean if exists(args.mean) else find_file(args.mean, cfg['RESOURCES_PATH_LIST'])

if 'LABELS_FILE_PATH' in cfg:
    import numpy as np
    labels = np.loadtxt(cfg['LABELS_FILE_PATH'], str, delimiter='\t')
else:
    labels = None

if cfg['USE_CPU']:
    import caffe
    caffe.set_mode_cpu()

# Project
from common.io import read_image, read_binproto_image
from common.validation import net_heatmap, net_forward_pass


def _display_classification_heatmap(image_path, model_path, weights_path, mean_image_path=None, labels=None, verbose=False):
    assert exists(image_path), "Input test image file is not found"
    assert exists(model_path), "Input model file is not found"
    assert exists(weights_path), "Input weights file is not found"

    image = read_image(image_path)
    mean_image = read_binproto_image(mean_image_path)
    if len(mean_image.shape) == 4:
        mean_image = mean_image.reshape(mean_image.shape[1:])
        if mean_image.shape[0] < mean_image.shape[1] or mean_image.shape[0] < mean_image.shape[2]:
            # (C, H, W) -> (H, W, C)
            mean_image = mean_image.transpose((1, 2, 0))

    # output = net_forward_pass(image=image, mean_image=mean_image, model_path=model_path, weights_path=weights_path)
    output, heatmap = net_heatmap(resolution=0.95, image=image, mean_image=mean_image,
                                  model_path=model_path, weights_path=weights_path)

    # # sort top five predictions from softmax output
    # top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
    # if labels is not None and len(labels) == len(output_prob):
    #     print len(labels), len(output_prob)
    #     print 'top probabilities and labels:', zip(output_prob[top_inds], labels[top_inds])
    # else:
    #     print 'top probabilities and indices:', zip(output_prob[top_inds], top_inds)
    #
    if verbose:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.show()

# _display_forward_pass(image_path, model_path, weights_path, mean_image_path=mean_image_path, labels=labels, verbose=args.verbose)
_display_classification_heatmap(image_path, model_path, weights_path, mean_image_path=mean_image_path, labels=labels, verbose=args.verbose)







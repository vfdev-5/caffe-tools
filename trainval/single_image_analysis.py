
"""
    Script to classify a single image

    Usage: single_image_analysis.py [-h] [--mean mean_image_file_path] [--full]
                                [--save-output SAVE_OUTPUT] [--verbose]
                                test_image_path model_file_path
                                weights_file_path


    positional arguments:
      test_image_path       Relative to DATASET_PATH or absolute path to the test
                            image
      model_file_path       Relative to one of MODELS_PATH_LIST or absolute path
                            to the model prototxt file
      weights_file_path     Relative to one of RESOURCES_PATH_LIST or absolute
                            path to the weights .caffemodel file

    optional arguments:
      -h, --help            show this help message and exit
      --mean mean_image_file_path
                            Relative to one of RESOURCES_PATH_LIST or absolute
                            path to the mean image .binproto file
      --full                Compute a full resolution heatmap
      --save-output SAVE_OUTPUT
                            Store computed results in the folder
      --verbose

"""

# Python
import argparse
import sys
from os import makedirs
from os.path import join, abspath, dirname, exists, isdir, basename

# Matplotlib
import matplotlib.pyplot as plt

# Setup common package path and CAFFE_PATH
COMMON_PACKAGE_PATH = abspath(join(dirname(abspath(__file__)), ".."))
sys.path.insert(0, COMMON_PACKAGE_PATH)
from common.config import cfg
sys.path.insert(0, join(cfg['CAFFE_PATH'], 'python'))

# import helping methods
from common import find_file, get_abspath


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
parser.add_argument('--full', action='store_true', help='Compute a full resolution heatmap')
parser.add_argument('--save-output', type=str, help='Store computed results in the folder')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

image_path = get_abspath(args.path)
model_path = get_abspath(args.model)
weights_path = get_abspath(args.weights)

assert image_path is not None, "Input test image file is not found"
assert model_path is not None, "Input model file is not found"
assert weights_path is not None, "Input weights file is not found"

mean_image_path = None
if args.mean is not None:
    mean_image_path = get_abspath(args.mean)
    assert mean_image_path is not None, "Input mean image file is not found"

output_folder = None
if args.save_output is not None:
    if not exists(args.save_output):
        makedirs(args.save_output)
    elif not isdir(args.save_output):
        raise Exception("The argument 'save-output' should not be a file")
    output_folder = abspath(args.save_output)

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
from common.validation import NetValidation, _blackboxed_image_iterator


def _display_classification_heatmap(image_path, model_path, weights_path, mean_image_path=None, labels=None,
                                    is_full=False, output_folder=None, verbose=False):

    nv = NetValidation(model_path, weights_path)
    nv.verbose = verbose
    nv.batch_size = 100

    image = read_image(image_path)
    mean_image = read_binproto_image(mean_image_path)
    if len(mean_image.shape) == 4:
        mean_image = mean_image.reshape(mean_image.shape[1:])
        if mean_image.shape[0] < mean_image.shape[1] or mean_image.shape[0] < mean_image.shape[2]:
            # (C, H, W) -> (H, W, C)
            mean_image = mean_image.transpose((1, 2, 0))

    nv.mean_image = mean_image

    # check net output type (the 1st one):
    key = nv.net.outputs[0]

    def _save_data(data, data_name):
        filename_output = join(output_folder, "%s_%s_%s.npy" %
                               (image_name, data_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        np.save(filename_output, output)

    if len(nv.net.blobs[key].data.shape) == 2:
        # Classes probability output -> can draw a heatmap
        if not is_full:
            output, heatmap = nv.tiled_heatmap(resolution=0.70, image=image)
        else:
            output, heatmap = nv.full_heatmap(bbox_size=40, image=image)

        if output_folder is not None:
            import datetime
            image_name = basename(image_path)
            _save_data(output, 'output')
            _save_data(heatmap, 'heatmap')

        output_prob = output[key][0]

        # sort top five predictions from softmax output
        top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
        if labels is not None and len(labels) == len(output_prob):
            print 'top probabilities and labels:', zip(output_prob[top_inds], labels[top_inds])
            plt.suptitle("Net detection : {} with probability {}".format(labels[top_inds[0]], output_prob[top_inds[0]]))
        else:
            print 'top probabilities and indices:', zip(output_prob[top_inds], top_inds)
            plt.suptitle("Net detection : {} with probability {}".format(top_inds[0], output_prob[top_inds[0]]))

        plt.subplot(1, 3, 1)
        plt.title("Original image")
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.title("Net features heatmap")
        plt.imshow(heatmap)
        merged = image.copy().astype(np.float64)
        merged[:, :, 0] *= heatmap
        merged[:, :, 1] *= heatmap
        merged[:, :, 2] *= heatmap
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("Merged images")
        plt.imshow(merged.astype(np.uint8))

        plt.show()

    else:
        # Launch forward pass and display the result:
        output = nv.forward_pass(image)
        output_image = output[key][0]

        if output_folder is not None:
            import datetime
            image_name = basename(image_path)
            _save_data(output, 'output')



        output_image = output_image.argmax(axis=0)

        plt.subplot(1, 2, 1)
        plt.title("Original image")
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title("Net output image")
        plt.imshow(output_image)
        plt.colorbar()

        plt.show()


_display_classification_heatmap(image_path, model_path, weights_path, mean_image_path=mean_image_path, labels=labels,
                                is_full=args.full, output_folder=output_folder, verbose=args.verbose)







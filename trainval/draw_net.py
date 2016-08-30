#
# """
#     Script to display the net
#
#     Usage:
#         python draw_net.py /path/to/model
#
#
#
# """
#
# # Python
# import argparse
# import os
# import sys
#
# # Numpy
# import numpy as np
#
# # Matplotlib
# import matplotlib.pyplot as plt
#
# # Setup common package path if executed as __main__
# if not __name__ == "__main__":
#     # Project
#     from common.config import cfg
#     # Caffe
#     import caffe
#     if cfg['USE_CPU']:
#         caffe.set_mode_cpu()
#
#
# def check(image_path, model_path, weights_path, verbose=False):
#     """
#     Compute models forward pass on the test image and display the results
#
#     :param image_path:
#     :param model_path:
#     :param weights_path:
#     :param verbose:
#     :return:
#     """
#
#     assert os.path.exists(image_path), "Input test image file is not found"
#     assert os.path.exists(model_path), "Input model file is not found"
#     assert os.path.exists(weights_path), "Input weights file is not found"
#
#     image = caffe.io.load_image(image_path)
#     if verbose:
#         plt.imshow(image)
#
#     net = caffe.Net(model_path, weights_path, caffe.TEST)
#
#
#
# if __name__ == "__main__":
#
#     # Setup common package path and CAFFE_PATH
#     from os.path import join, abspath, dirname
#
#     COMMON_PACKAGE_PATH = abspath(join(dirname(abspath(__file__)), ".."))
#     sys.path.insert(0, COMMON_PACKAGE_PATH)
#     from common.config import cfg
#     from common import find_file
#     sys.path.insert(0, join(cfg['CAFFE_PATH'], 'python'))
#     import caffe
#     if cfg['USE_CPU']:
#         caffe.set_mode_cpu()
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('path', metavar='test_image_path', type=str, help='Relative to DATASET_PATH or absolute path to the test image')
#     parser.add_argument('model', metavar='model_file_path', type=str, help='Relative to one of MODELS_PATH_LIST or absolute path to the model file')
#     parser.add_argument('weights', metavar='weights_file_path', type=str, help='Relative to one of WEIGHTS_PATH_LIST or absolute path to the weights file')
#     parser.add_argument('--verbose', action='store_true')
#     args = parser.parse_args()
#
#     image_path = args.path if os.path.exists(args.path) else find_file(args.path, [cfg['DATASET_PATH']])
#     model_path = args.model if os.path.exists(args.model) else find_file(args.model, cfg['MODELS_PATH_LIST'])
#     weights_path = args.weights if os.path.exists(args.weights) else find_file(args.weights, cfg['WEIGHTS_PATH_LIST'])
#
#     check(image_path, model_path, weights_path, verbose=args.verbose)

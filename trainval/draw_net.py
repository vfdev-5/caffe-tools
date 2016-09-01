
#
# """
#     Script to display the net
#
#   usage: draw_net.py [-h] [--direction {TB,LR}] [--phase {TEST,TRAIN}]
#                    models [models ...]
#
#   positional arguments:
#       models                List of models. Pathes can be absolute or relative to
#                             MODELS_PATH_LIST. For example, /path/to/model1
#                             [model2] [...].
#
#   optional arguments:
#       -h, --help            show this help message and exit
#       --direction {TB,LR}   Graph layout direction: top-bottom (TB), left-rigth
#                         (LR)
#       --phase {TEST,TRAIN}  Network phase: train or test. If not specified phase
#                         is ALL
#
# """


# Python
import argparse
import sys
from os.path import join, abspath, dirname, exists, isdir, basename

# Setup common package path and CAFFE_PATH
COMMON_PACKAGE_PATH = abspath(join(dirname(abspath(__file__)), ".."))
sys.path.insert(0, COMMON_PACKAGE_PATH)
from common.config import cfg
sys.path.insert(0, join(cfg['CAFFE_PATH'], 'python'))

from google.protobuf import text_format
from caffe import TRAIN, TEST
from caffe.draw import draw_net
from caffe.proto import caffe_pb2

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Project
from common import get_abspath


parser = argparse.ArgumentParser()
parser.add_argument("models", nargs='+', type=str,
                    help="List of models. Pathes can be absolute or relative to MODELS_PATH_LIST. For example, /path/to/model1 [model2] [...]. ")
parser.add_argument("--direction", type=str, choices=['TB', 'LR'],
                    help="Graph layout direction: top-bottom (TB), left-rigth (LR)")
parser.add_argument("--phase", type=str, choices=['TEST', 'TRAIN'],
                    help="Network phase: train or test. If not specified phase is ALL")


args = parser.parse_args()


for model_path in args.models:
    model_abspath = get_abspath(model_path)
    assert model_abspath is not None, "Model file '%s' is not found" % model_path

    print model_path, model_abspath
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(model_abspath).read(), net)
    phase = None if args.phase is None else TRAIN if 'TRAIN' in args.phase else TEST
    rankdir = args.direction if args.direction is not None else 'TB'

    output = draw_net(net, rankdir, ext='png', phase=phase)

    graph_img = np.asarray(bytearray(output), dtype=np.uint8)
    graph_img = cv2.imdecode(graph_img, 0)

    plt.figure()
    plt.title("Net graph '%s'" % model_path)
    plt.imshow(graph_img, cmap='gray', aspect='auto')
    plt.axis('off')

plt.show()

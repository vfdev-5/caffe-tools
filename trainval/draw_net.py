
#
# """
#     Script to display the net
#
#   usage: draw_net.py [-h] [--direction {TB,LR}] [--phase {TEST,TRAIN}]
#                    model [model ...]
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
from os import environ
from os.path import join, abspath, dirname, exists, isdir, basename

# Setup common package path and CAFFE_PATH
COMMON_PACKAGE_PATH = abspath(join(dirname(abspath(__file__)), ".."))
sys.path.insert(0, COMMON_PACKAGE_PATH)
from common.config import cfg
sys.path.insert(0, join(cfg['CAFFE_PATH'], 'python'))

environ['GLOG_minloglevel'] = '3' 
from google.protobuf import text_format
from caffe import TRAIN, TEST, Net
from caffe.draw import draw_net
from caffe.proto import caffe_pb2
environ['GLOG_minloglevel'] = '1'

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

def print_net_info(model_path, phase, net_params):
    # disable caffe logging
    net = Net(model_path, phase=phase)
    
    # Blobs = Data, net_params.layer = Layers <=> wont be equal
    
    nb_layers = len(net_params.layer)
    print "Net architecture : ", model_path
    print " - layer name (layer type) : (batch size, feature dim, width, height) | number of filters, (filter dim, filter width, filter height), stride, padding"
    counter=0
    for k, v in net.blobs.items():
        params = []
        if k in net.params:
            params = net.params[k][0].data.shape
            padding = None
            stride = None
            layer_type = None
            if nb_layers > 0:
                layer = net_params.layer[counter]
                layer_type = layer.type
                if layer_type == 'Convolution' or layer_type == 'Deconvolution':
                    stride = layer.convolution_param.stride[0] if len(layer.convolution_param.stride._values) else 1
                    padding = layer.convolution_param.pad[0] if len(layer.convolution_param.pad._values) else 0
                elif layer_type == 'Pooling':
                    stride = layer.pooling_param.stride
                    padding = layer.pooling_param.pad
                
            if padding and stride and layer_type is not None:
                print "%s (%s) : %s | %s, %s, stride=%s, padding=%s" %(k, layer_type, v.data.shape, params[0], params[1:], stride, padding) 
            else:
                print "%s : %s | %s, %s" %(k, v.data.shape, params[0], params[1:]) 
        else:
            print "%s : %s" %(k, v.data.shape) 
        counter += 1 
    
    
can_draw=False

for model_path in args.models:
    model_abspath = get_abspath(model_path)
    assert model_abspath is not None, "Model file '%s' is not found" % model_path

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(model_abspath).read(), net)
    phase = TEST if args.phase is None else TRAIN if 'TRAIN' in args.phase else TEST
    rankdir = args.direction if args.direction is not None else 'TB'
    
    can_draw = len(net.layer) > 0
    output = draw_net(net, rankdir, ext='png', phase=phase)

    graph_img = np.asarray(bytearray(output), dtype=np.uint8)
    graph_img = cv2.imdecode(graph_img, 0)

    if can_draw:
        plt.figure()
        plt.title("Net graph '%s'" % model_path)
        plt.imshow(graph_img, cmap='gray', aspect='auto')
        plt.axis('off')
        
    else:
        print "!!! Version of the model '%s' is too old and can not be drawn as a graph" % model_path
        
    print_net_info(model_abspath, phase, net)

if can_draw:
    plt.show()

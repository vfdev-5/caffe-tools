

#
# Script to setup caffe distribution path
#

import sys
from os.path import join, abspath, dirname

cfg = {}


def _configure():
    global cfg
    filename = abspath(join(dirname(abspath(__file__)), "..", "config.yaml"))
    import yaml
    with open(filename, 'r') as f:
        cfg = yaml.load(f)

_configure()

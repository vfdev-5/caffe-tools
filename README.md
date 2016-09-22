# A set of tools to use with [BVLC/caffe](https://github.com/BVLC/caffe) for deep learning with images

All tools are written in python and inspired from ideas and algorithms found in various sources.

Tools are divided into categories:

- train/test data preprocessing  
- training / validation / visualization

### Train/test data preprocessing  

- ...
- ...
- ...

### Training / validation / visualization

- single image validation and features visualization for classification CNN

  - `single_image_analyze.py` computes net forward pass on a single image and a 'heatmap' of trained features of the model detected on this image. See [here](https://github.com/vfdev-5/caffe-tools/blob/master/trainval/README.md) for more details.


For example, features 'heatmap' of the [famous cat](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb) looks like

<img src="https://github.com/vfdev-5/caffe-tools/blob/master/examples/results/ref_caffenet_cat_visu.png" width="1000"/>

```
python trainval/single_image_analysis.py cat.jpg bvlc_reference_caffenet/deploy.prototxt bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --mean ilsvrc12/imagenet_mean.binaryproto --verbose
```

- network model's graph visualization with matplotlib

  - `draw_net.py` displays network model graph as figures using matplotlib. Make sure that you have *graphviz* and *pydotplus* installed.
  
  ```
  python draw_net.py /path/to/model.prototxt relative/path/to/model2.prototxt
  ```

## Python 2.7 and dependencies :

* pycaffe from [BVLC/caffe](https://github.com/BVLC/caffe)
* PyYAML
* numpy
* matplotlib
* scipy
* sklearn
* pandas
* opencv
* lmdb


~~See `requirements.txt` file~~


## How to use

Setup your configuration in the `config.yaml`:  
(you can rename and modify `config.yaml.example`)
```
# Path to Caffe repository
CAFFE_PATH: /path/to/caffe/

# Path to image dataset
DATASET_PATH: /Pascal_VOC_2011/VOCdevkit/VOC2011/JPEGImages/

# Path to labels file
LABELS_FILE_PATH: /path/to/data/ilsvrc12/synset_words.txt

# List of available nets: [ [<model_label>, <model_path>, <weights_path or ''>, <mean_image_path or ''>], ... ]
NETS_LIST: {
  'caffe_net': ['bvlc_reference_caffenet/deploy.prototxt', 'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 'ilsvrc12/imagenet_mean.binaryproto'],
  'vgg16': ['vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'vgg/VGG_ILSVRC_16_layers.caffemodel', 'ilsvrc12/imagenet_mean.binaryproto'],
  'resnet50': ['ResNet/ResNet-50-deploy.prototxt', 'ResNet/ResNet-50-model.caffemodel', 'ilsvrc12/imagenet_mean.binaryproto']
}

# List with paths to models
MODELS_PATH_LIST: [
  /path/to/caffe/models,
  /path/to/fcn/voc-fcn16s
]

# List with paths to weights
RESOURCES_PATH_LIST: [
  /path/to/models,
  /path/to/fcn/voc-fcn16s
]


# Use CPU instead of GPU
USE_CPU: false

```

## Scripts of the package 'trainval'

Before launching scripts, make sure that `config.yaml` file is properly setup.

####  `single_image_analysis.py`

Script, at first, sets up a trained net, compute forward pass and display the result. Second, it computes model's detected features as a 'heatmap'. Computed heatmap image visually looks like [CAM results](https://github.com/metalbubble/CAM), however the method is different. There are two options to compute the heatmap: *approximative* and *full*.

Usage :
```
python single_image_analysis.py <image_path> <model_prototxt_path> <weights_path> [--mean <mean_image_path>] [--verbose] [--full]
```
- Input image path <image_path> can be absolute or a relative to DATASET_PATH (see `config.yaml`).
- Net model file path <model_prototxt_path> can be absolute or a relative to one of MODELS_PATH_LIST entries.
- Path of the model's trained weights <weights_path> can be absolute or a relative to one of RESOURCES_PATH_LIST entries.
- Turn on verbose option with `--verbose`. There is a heatmap computation window is shown to observe intermediate result.
- Compute the heatmap in full resolution. **Attention, it could be extremely time consuming**


For example :

```
python single_image_analysis.py cat.jpg bvlc_reference_caffenet/deploy.prototxt bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --mean ilsvrc12/imagenet_mean.binaryproto --verbose
```

The method to compute model's detected features as a 'heatmap' is simply to compute net forward passes on the input image with a black box inserted at some location and moving it in the image [1][1]. Each forward pass output is compared with the initial one and a scalar product of these two outputs is computed and written to the heatmap.



___

[1]: My reference on this method is from [here](https://habrahabr.ru/post/307078/)

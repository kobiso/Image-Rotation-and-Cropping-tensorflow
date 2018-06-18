# Image Rotation and Cropping in TensorFlow
This is an implementation and visualization of image rotation and cropping out black borders in TensorFlow.
TensorFlow support only [image rotation function](https://www.tensorflow.org/api_docs/python/tf/contrib/image/rotate) `tf.contrib.image.rotate(images, angles, interpolation, name)`.
However, when you rotate an image with this function, there will be black noise on each border as below.

![Goal](/images/example1.png)

So, we want to cropping out this black borders in TensorFlow, especially when the image is loaded as Tensor and it has to go through preprocessing phase.
The implementation include example and visualization with [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/).

## Core Functions
If you do not want to run the code or see the visualization, you can just copy and paste the core functions.
In [*read_tfrecord.py* file](https://github.com/kobiso/Image-Rotation-and-Cropping-tensorflow/blob/master/read_tfrecord.py), `_rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop)` and `_largest_rotated_rect(w, h, angle)` are core functions.

## Prerequisites
- Python 3.4+
- TensorFlow 1.5+
- Jupyter Notebook
- Python packages: requirements.txt
  - Simply install it by running : `pip install -r /path/to/requirements.txt` in the shell

## Prepare the Tiny ImageNet
Download the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) in this link and unzip it.
Set the path of the dataset on variable `TINY_IMAGENET_DIRECTORY` in `build_tfrecords.ipynb` file.

## Convert to TFRecords
As test set does not include class labels and bounding boxes, validation set will be used as test set in this implementation.
And training set will be divided with certain percentage (as you defined) into training set and validation set.
Each data set (training, validation and test) will have iamges, labels and bounding box information.

To convert Tiny ImageNet to TFRecords, set each requiring path in `build_tfrecords.ipynb` and run all cell.
Then TFRecords files will be created in the designated path you defined.
Note that you can set the validation ratio in the variable `VALIDATION_RATIO`.

## Visualize Original, Rotated and Cropped Image
You can check and visualize TFRecords file in `check_tfrecords.ipynb`.
In `read_tfrecord.read_tfrecord()` function, you can set `rotation_degree` and `do_crop` arguments to rotate and crop images.

- **Original Image**

![Example1](/images/origin.png)

- **Rotated Image**

![Example2](/images/rotation.png)

- **Rotated and Cropped Image**

![Example3](/images/crop.png)

## Reference
- StackOverflow: [Rotate image and crop out black borders](https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders)
- [TensowFlow example code for converting ImageNet data to TFRecords file format](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py)
  
## Author
Byung Soo Ko / kobiso62@gmail.com

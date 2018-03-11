# Tiny-ImageNet-to-TFRecords
This is an implementation to convert [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset from Standford CS231N to [TFRecords](https://www.tensorflow.org/programmers_guide/datasets).

Tiny Imagenet has 200 classes.
Each class has 500 training images, 50 validation images, and 50 test images.
Training images and validation images have annotations including class labels and bounding boxes.
But the test set does not include either class labels or bounding boxes.
Considering this aspect, this implementation cab be used in two ways: for **classification** and for **object localization**.

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

## Check and Visualize TFRecords File
You can check and visualize TFRecords file with `check_tfrecords.ipynb`.
After reading the TFRecords, the data will be saved in `read_data_dict` dictionary.

- **Example output of visualization**

![Example](/images/tfrecords_example.png)
  
## Reference
- [TensowFlow example code for converting ImageNet data to TFRecords file format](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py)
  
## Author
Byung Soo Ko / kobiso62@gmail.com

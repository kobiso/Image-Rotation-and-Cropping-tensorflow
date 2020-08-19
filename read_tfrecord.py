"""This file is to read tfrecord data for training and test phrase
"""

import os
import pickle
import random
import tensorflow as tf
import numpy as np

import math

def read_tfrecord(tfrecord_train_dir, tfrecord_valid_dir, tfrecord_test_dir, batch_size=64, shuffle_buffer_size=5000, rotation_degree=0, do_crop=False):    
    tfrecord_dir_dict = {'train': tfrecord_train_dir, 'valid': tfrecord_valid_dir, 'test': tfrecord_test_dir}
    tfrecord_list_dict = {}
    dataset_dict = {}
    data_initializer_dict = {}
    return_dict = {}    
  
    for data_type in ['train', 'valid', 'test']:
        tfrecord_list_dict[data_type] = [os.path.join(tfrecord_dir_dict[data_type], dir_) 
                                    for dir_ in os.listdir(tfrecord_dir_dict[data_type])]
        dataset_dict[data_type] = create_tiny_image_dataset(path=tfrecord_list_dict[data_type],
                                                            batch_size=batch_size,
                                                            prefetch_buffer_size=shuffle_buffer_size,
                                                            repeat=False,
                                                            shuffle=True,
                                                            rotation_degree=rotation_degree,
                                                            do_crop=do_crop)

        if data_type == 'train':
            iter_ = tf.data.Iterator.from_structure(dataset_dict[data_type].output_types, dataset_dict[data_type].output_shapes)
        data_initializer_dict[data_type] = iter_.make_initializer(dataset_dict[data_type])
        return_dict['data_initializer_' + data_type] = data_initializer_dict[data_type]
        
    X, Loc, Y_one_hot = iter_.get_next()
    return_dict.update({'X':X, 'Loc':Loc, 'Y_one_hot': Y_one_hot})
    
    return return_dict

def create_tiny_image_dataset(path, batch_size=64, prefetch_buffer_size=5000, repeat=False, shuffle=False, rotation_degree=0, do_crop=False):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_example_proto)
    dataset = dataset.map(lambda x,y,z: preprocess_image(image=x, location=y, label_one_hot=z, height=224, width=224, rotation_degree=rotation_degree, do_crop=do_crop))
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(buffer_size=prefetch_buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset

def preprocess_image(image, location, label_one_hot, height=224, width=224, rotation_degree=0, do_crop=False):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would cropt the central fraction of the
    input image.

    Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
    Returns:
    3-D float Tensor of prepared image.
    """

    #if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    #if central_fraction:
    #  image = tf.image.central_crop(image, central_fraction=central_fraction)
    
    image = _rotate_and_crop(image, height, width, rotation_degree, do_crop)

    #if height and width:
    # Resize the image to the specified height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])

    #image = tf.cast(image, tf.float32)
    #image = tf.multiply(image, 1/255.)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0) 
            
    return image, location, label_one_hot

def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

    Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

    Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
            'height': tf.FixedLenFeature((), tf.int64),
            'width': tf.FixedLenFeature((), tf.int64),
            'channel': tf.FixedLenFeature((), tf.int64),
            'label': tf.FixedLenFeature((), tf.int64),
            'label_depth': tf.FixedLenFeature((), tf.int64),
            'label_one_hot_raw': tf.FixedLenFeature((), tf.string),
            'image_raw': tf.FixedLenFeature((), tf.string),
            'location_raw': tf.FixedLenFeature((), tf.string)}
    
    #sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.

    features = tf.parse_single_example(example_serialized, feature_map)
    
    image_raw = tf.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image_raw, [64, 64, 3])
    label = tf.cast(features['label'], dtype=tf.int32)
    label_one_hot = tf.decode_raw(features['label_one_hot_raw'], tf.float64)
    location = tf.decode_raw(features['location_raw'], tf.int64)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    #bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    #bbox = tf.expand_dims(bbox, 0)
    #bbox = tf.transpose(bbox, [0, 2, 1])

    return image, location, label_one_hot

def _rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """Rotate the given image with the given rotation degree and crop for the black edges if necessary
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.

    Returns:
        A rotated image.
    """
  
    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:
        image = tf.contrib.image.rotate(image, math.radians(rotation_degree), interpolation='BILINEAR')
      
        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = _largest_rotated_rect(output_width, output_height, math.radians(rotation_degree))
            resized_image = tf.image.central_crop(image, float(lrr_height)/output_height)    
            image = tf.image.resize_images(resized_image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    
    return image

def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )
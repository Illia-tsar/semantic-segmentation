import os
import numpy as np
import tensorflow as tf
from imageio.v2 import imread
from functools import partial
from constants import *
from albumentations import Compose, HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, RandomSizedCrop


def rle_encode(mask):
    """
    Run-Length Encoding
    
    Arguments:
        mask -- numpy array of shape (height, width, 1)
    Returns:
        string with encoded pixels
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join([str(x) for x in runs])


def rle_decode(rle):
    """
    Run-Length Decoding
    
    Arguments:
        rle -- string with encoded pixels
    Returns:
        numpy array of shape (height, width, 1)
    """
    shape = (768, 768)
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    rle_mask = rle.split()

    # elements with uneven indices are related to start idx
    # elements with even indices are related to lengths
    idx, length = [np.array(x, dtype=int) for x in (rle_mask[::2], rle_mask[1::2])]
    for a, b in zip(idx-1, idx+length-1):
        mask[a:b] = 1
    return mask.reshape(shape).T[..., np.newaxis]


def rle_object_size(rle):
    """
    Calculate object size in pixels
    
    Arguments:
        rle -- string with encoded pixels
    Returns:
        int -- quantity of pixels in object
    """
    rle_mask = rle.split()
    # take every second element of array and calculate the sum of obtained list
    object_size = np.sum([int(x) for x in rle_mask[1::2]])
    return object_size


def apply_mask(img, rle):
    """
    Apply mask to img
    
    Arguments:
        img -- numpy array of shape (height, width, 3)
        rle -- string with encoded pixels
    Returns:
        img -- numpy array with mask applied to it of shape (height, width, 3)
    """
    mask = rle_decode(rle)
    return img * mask[:, :, np.newaxis]


def aug_fn(image, mask, img_size):
    """
    Helper function -- performs augmentation

    Arguments:
        image -- tf.Tensor of shape (height, width, 3)
        mask -- tf.Tensor of shape (height, width, 1)
        img_size -- size of an image
    Returns:
        Augmented image and mask of the same sizes as the input ones
    """
    transforms = Compose([
        RandomSizedCrop(min_max_height=(int(0.89*img_size), img_size), height=img_size, width=img_size, p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5)
    ])
    data = {'image': image, 'mask': mask}
    aug_data = transforms(**data)

    aug_img = aug_data['image']
    aug_mask = aug_data['mask']

    return aug_img, aug_mask


def process_img(image, mask, img_size):
    """
    Function to pass data in through the pipeline
    Wraps the helper function with tf.numpy_function

    Arguments:
        image -- tf.Tensor of shape (height, width, 3)
        mask -- tf.Tensor of shape (height, width, 1)
        img_size -- size of an image
    Returns:
        Augmented image and mask of the same sizes as the input ones
    """
    aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, mask, img_size], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask


def _fixup_shape(x, y):
    """
    The mapping used right after batching
    In some cases inferring of the shapes of the output Tensor may fail due to the use of .from_generator()

    Arguments:
        x, y -- tf.Tensor
    Returns:
        tuple(tf.Tensor, tf.Tensor)
    """
    x.set_shape([None, None, None, 3])
    y.set_shape([None, None, None, 1])
    return x, y


def transform_dataset(img_data, img_size):
    """
    Helper function -- reads, normalizes and resizes image, decodes and resizes mask

    Arguments:
        img_data -- string of structure 'image_id+encoded_pixels'
        img_size -- size to resize image and mask to
    Returns:
        tuple(tf.Tensor, tf.Tensor)
    """
    data = img_data.numpy().decode('utf-8')
    img_id, enc_pix = data.split('+')

    img = imread(os.path.join(TRAIN_IMG_DIR, img_id))
    mask = rle_decode(enc_pix)

    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    return tf.image.resize(img, size=[img_size, img_size]), tf.image.resize(mask, size=[img_size, img_size])


def load_dataset(img_series, img_size):
    """
    Function creating tf.data.Dataset generator with parallelized processing

    Arguments:
        img_series -- pd.Series filled with rle encoded masks and indexed with image ids
        img_size -- size to resize image and mask to
    Returns:
        tf.data.Dataset
    """
    # creating list of strings of structure 'image_id+encoded_pixels'
    data_gen = ['+'.join([img_id, enc_pix]) for img_id, enc_pix in zip(img_series.index, img_series)]
    # creating Dataset from lightweight generator
    dataset = tf.data.Dataset.from_generator(lambda: data_gen, tf.string)

    dataset = dataset.shuffle(buffer_size=len(data_gen), reshuffle_each_iteration=True)
    # convert datapoint of struct 'image_id+encoded_pixels' to tuple(image: tf.Tensor, mask: tf.Tensor)
    dataset = dataset.map(lambda img_data: tf.py_function(func=transform_dataset, inp=[img_data, img_size],
                                                          Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def prepare_dataset(dataset, img_size, augment=False):
    """
    Processing function -- optionally applies augmentation and batches dataset

    Arguments:
        dataset -- tf.data.Dataset
        img_size -- size of an image
    Returns:
        batched dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if augment:
        dataset = dataset.map(partial(process_img, img_size=img_size), num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

    dataset = dataset.repeat().batch(BATCH_SIZE).map(_fixup_shape, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

    return dataset

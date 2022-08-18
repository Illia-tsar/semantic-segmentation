import numpy as np
import pandas as pd
import tensorflow as tf
from imageio.v2 import imread
from utils import rle_decode
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cv2 import resize
from PIL import Image
from model import unet_model
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle
from keras import models
from keras.utils import plot_model
import matplotlib.pyplot as plt


def image_generator(img_names, batch_size, shape=(768, 768)):
    """
    Image Generator

    Arguments:
        img_names -- pandas series with image ids
        batch_size -- int denoting the batch size
        shape -- tuple denoting the dims of output image
    Returns:
        tuple combined of normalized images and masks
    """
    batches = img_names.reset_index().to_numpy()    # image ids
    while True:
        np.random.shuffle(batches)
        for idx in range(img_names.shape[0] // batch_size):
            batch_img = batches[idx * batch_size: (idx + 1) * batch_size, 0]    # get #batch_size img ids
            batch_masks = batches[idx * batch_size: (idx + 1) * batch_size, 1]  # get #batch_size encoded image masks

            imgs = [resize(imread(os.path.join(TRAIN_IMG_DIR, img_id)), shape)
                    for img_id in batch_img]    # read images by ids and resize them to the given shape
            masks = [np.array(Image.fromarray(rle_decode(msk)).resize(shape))[:, :, np.newaxis]
                     for msk in batch_masks]    # read masks by ids and resize them to the given shape
            yield np.array(imgs) / 255.0, np.array(masks, dtype=float)


def augmenter(gen):
    """
    Image Augmenter

    Arguments:
        gen -- image generator
    Returns:
         tuple combined of normalized augmented images and masks
    """
    image_gen = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last')
    for imgs, msks in gen:
        seed = np.random.choice(range(9999))    # use the same seed for image and mask to keep them in sync

        aug_img = image_gen.flow(255 * imgs,
                                 batch_size=imgs.shape[0],
                                 seed=seed,
                                 shuffle=True)
        aug_msk = image_gen.flow(msks,
                                 batch_size=imgs.shape[0],
                                 seed=seed,
                                 shuffle=True)
        yield next(aug_img) / 255.0, next(aug_msk)


def dice_score(y_true, y_pred, smooth=1):
    """
    Dice Score metric

    Calculates the Dice Score as 2 * intersection / union
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred[..., tf.newaxis], float)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3]) # intersection between predicted and true values
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) # union between predicted and true values
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def IoU(y_true, y_pred):
    """
    IoU metric

    Calculates Intersection over Union as intersection / (union - intersection)
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred[..., tf.newaxis], float)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean(intersection / (union - intersection + 1.0), axis=0)


def train(model=None, shape=(768, 768), plot_model_struct=False, save_history=True):
    """
    Main training function

    Main steps:
    1.Get masks
    2.Filter out no-ships images
    3.Split data into training and validation sets
    4.Get data augmenting generator for training set and image generator for validation set
    5.Compile and fit the model

    Arguments:
        shape -- the shape of an image
        model -- U-Net model
        plot_model_struct -- boolean denoting whether to plot model struct or not
        save_history -- boolean denoting whether to store history or not
    Returns:
        None
    """
    masks = pd.read_csv(SEGMENTATION, delimiter=',')
    # drop all no-ships images
    masks.dropna(axis=0, inplace=True)
    # combine encoded pixels gathered by image ids
    masks = masks.groupby(by='ImageId').EncodedPixels.apply(lambda x: ' '.join(x))
    # split the data into training and validation sets
    train_names, val_names = train_test_split(masks, test_size=0.1, random_state=17)

    train_batch_gen = augmenter(image_generator(train_names, BATCH_SIZE, shape)) # train batch generator
    val_batch_gen = image_generator(val_names, BATCH_SIZE, shape)   # validation batch generator

    if model is None:
        model = unet_model(input_size=(shape[0], shape[1], 3), n_filters=8, n_classes=2)    # init U-Net

    model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[dice_score, IoU])

    # save the best model during training
    checkpoint = ModelCheckpoint(SAVE_MODEL, monitor='val_dice_score', verbose=1, save_best_only=True,
                                 mode='max', save_weights_only=False)

    # control learning rate decay
    lr_decay = ReduceLROnPlateau(monitor='val_dice_score', verbose=1, mode='max', patience=3,
                                 cooldown=2, factor=0.6, min_lr=1e-5)

    # handle overfitting
    stopping = EarlyStopping(monitor='val_dice_score', min_delta=1e-5, mode='max', patience=10)
    callbacks = [checkpoint, lr_decay, stopping]

    train_steps_epoch = min(MAX_STEPS_TRAIN, train_names.shape[0] // BATCH_SIZE)
    val_steps_epoch = min(MAX_STEPS_VAL, val_names.shape[0] // BATCH_SIZE)

    model_history = model.fit_generator(train_batch_gen,
                                        steps_per_epoch=train_steps_epoch,
                                        epochs=EPOCHS,
                                        verbose=1,
                                        validation_data=val_batch_gen,
                                        validation_steps=val_steps_epoch,
                                        callbacks=callbacks)

    if save_history:
        with open(SAVE_HISTORY, 'wb') as f:
            pickle.dump(model_history.history, f)  # dump history data for future visualization

    if plot_model_struct:
        plot_model(model, to_file=PLOT_MODEL_PATH, show_shapes=True,
                   show_dtype=True, show_layer_activations=True)


def plot_history():
    """
    Main plotting function
    It displays plots split by metrics(loss, dice score, IoU) for history dumps listed in HISTORY_TO_PLOT
    The function displays curves based on values gained assessing model's performance on training and validation sets
    """
    history = []
    loss, val_loss = [], []
    dice_sc, val_dice_sc = [], []
    iou, val_iou = [], []

    # load history dicts
    for h in HISTORY_TO_PLOT:
        with open(h, 'rb') as f:
            history.append(pickle.load(f))

    _, ax = plt.subplots(1, 3, figsize=(14, 12)) # create 3 subplots(3 metrics: loss, dice score, IoU)

    # add values to declared lists
    for row in history:
        loss.extend(row['loss'])
        val_loss.extend(row['val_loss'])

        dice_sc.extend(row['dice_score'])
        val_dice_sc.extend(row['val_dice_score'])

        iou.extend(row['IoU'])
        val_iou.extend(row['val_IoU'])

    # plot the data
    with plt.xkcd():
        ax[0].plot(range(len(loss)), loss, label='loss')
        ax[0].plot(range(len(val_loss)), val_loss, label='val_loss')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(range(len(dice_sc)), dice_sc, label='dice_score')
        ax[1].plot(range(len(val_dice_sc)), val_dice_sc, label='val_dice_score')
        ax[1].legend()
        ax[1].grid(True)

        ax[2].plot(range(len(iou)), iou, label='IoU')
        ax[2].plot(range(len(val_iou)), val_iou, label='val_IoU')
        ax[2].legend()
        ax[2].grid(True)
        plt.show()


if __name__ == '__main__':
    # enable memory growth so that runtime initialization don't allocate all memory on the device
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # constants
    TRAIN_IMG_DIR = '../data/train_v2/'    # training images path
    SEGMENTATION = '../data/train_ship_segmentations_v2.csv'    # training images segmentations path
    SAVE_MODEL = '../model/final_model.h5'  # model saving path
    LOAD_MODEL = '../model/final_model.h5'    # path for model loading
    SAVE_HISTORY = '../model/history/history2.pickle'   # history saving path
    PLOT_MODEL_PATH = '../model/model.png'  # path specifies where to save generated model plot
    # plot history for training model on 192x192 images for 10 epochs and 384x384 images for 5 epochs
    HISTORY_TO_PLOT = ['../model/history/history1.pickle', '../model/history/history2.pickle']
    EPOCHS = 1
    BATCH_SIZE = 4
    MAX_STEPS_TRAIN = 800
    MAX_STEPS_VAL = 800

    plot_history()

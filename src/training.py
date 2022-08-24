import pandas as pd
import tensorflow as tf
from utils import load_dataset, prepare_dataset
from sklearn.model_selection import train_test_split
from model import unet_model
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle
from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import BinaryIoU
from constants import *


def dice_score(y_true, y_pred, smooth=1):
    """
    Dice Score metric

    Arguments:
        y_true -- true masks of shape (batch_size, height, width, 1)
        y_pred -- predicted probabilities of shape (batch_size, height, width, 1)
        smooth -- parameter to avoid zero division
    Returns:
        Dice score focused on class 1
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def create_model(n_filters=8, plot_model_struct=False):
    """
    Model creating function

    Arguments:
        n_filters -- number of filters to start with(further layers will have [n_filters * 2**n] filters)
        plot_model_struct -- boolean denoting whether to plot model struct or not
    Returns:
        tf.keras.Model
    """
    model = unet_model(input_size=(None, None, 3), n_filters=n_filters)

    # we use dice_score as evaluation metric to estimate model's confidence at predicting class 1
    # BinaryIoU is used to observe model's accuracy predicting classes 0 and 1 and is calculated as: (IoU_1 + IoU_0) / 2.0
    model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[dice_score, BinaryIoU(threshold=0.5)])

    if plot_model_struct:
        plot_model(model, to_file=PLOT_MODEL_PATH, show_shapes=True,
                   show_dtype=True, show_layer_activations=True)

    return model


def train(model, img_size=768, save_history=True):
    """
    Main training function

    Main steps:
    1.Get masks
    2.Filter out no-ships images
    3.Split data into training and validation sets
    4.Compile and fit the model

    Arguments:
        img_size -- the size of an image
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

    # load images and corresponding masks
    train_ds = load_dataset(train_names, img_size=img_size)
    val_ds = load_dataset(val_names, img_size=img_size)

    # augment and resize obtained datasets
    train_ds = prepare_dataset(train_ds, img_size=img_size, augment=True)
    val_ds = prepare_dataset(val_ds, img_size=img_size)

    # save the best model during training
    checkpoint = ModelCheckpoint(SAVE_MODEL, monitor='val_dice_score', verbose=1, save_best_only=True,
                                 mode='max', save_weights_only=True)

    # control learning rate decay
    lr_decay = ReduceLROnPlateau(monitor='val_dice_score', verbose=1, mode='max', patience=3,
                                 cooldown=2, factor=0.5, min_lr=1e-5)

    # handle overfitting
    stopping = EarlyStopping(monitor='val_dice_score', min_delta=1e-5, mode='max', patience=4)
    callbacks = [checkpoint, lr_decay, stopping]

    train_steps_epoch = min(MAX_STEPS_TRAIN, train_names.shape[0] // BATCH_SIZE)
    val_steps_epoch = min(MAX_STEPS_VAL, val_names.shape[0] // BATCH_SIZE)

    model_history = model.fit(train_ds,
                              steps_per_epoch=train_steps_epoch,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=val_ds,
                              validation_steps=val_steps_epoch,
                              callbacks=callbacks)

    if save_history:
        with open(SAVE_HISTORY, 'wb') as f:
            pickle.dump(model_history.history, f)  # dump history data for future visualization


def plot_history():
    """
    Main plotting function
    It displays plots split by metrics(loss, dice score, BinaryIoU) for history dumps listed in HISTORY_TO_PLOT
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

    _, ax = plt.subplots(1, 3, figsize=(14, 12)) # create 3 subplots(3 metrics: loss, dice score, BinaryIoU)

    # add values to declared lists
    for row in history:
        loss.extend(row['loss'])
        val_loss.extend(row['val_loss'])

        dice_sc.extend(row['dice_score'])
        val_dice_sc.extend(row['val_dice_score'])

        iou.extend(row['binary_io_u'])
        val_iou.extend(row['val_binary_io_u'])

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

        ax[2].plot(range(len(iou)), iou, label='binary_io_u')
        ax[2].plot(range(len(val_iou)), val_iou, label='val_binary_io_u')
        ax[2].legend()
        ax[2].grid(True)
        plt.show()


if __name__ == '__main__':
    # enable memory growth so that runtime initialization don't allocate all memory on the device
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    plot_history()

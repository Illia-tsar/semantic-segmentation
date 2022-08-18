import numpy as np
import pandas as pd
from imageio.v2 import imread
from utils import rle_encode
import os
import tensorflow as tf
from keras import models
from tqdm import tqdm


def write_result():
    """
    Write result to csv

    Main steps:
    1.Load model specified in LOAD_MODEL
    2.Get all test images names
    3.Read and normalize images one by one
    4.For each image:
        4.1.Get prediction of shape (1, height, width, n_classes=2)
        4.2.Map probabilities of last dimension to class labels
        4.3.Encode mask and append to submission dict
    """
    model = models.load_model(LOAD_MODEL, compile=False) # load model specified in MODEL_PATH constant
    test_names = os.listdir(TEST_IMG_DIR)    # get test images names

    submission = {'ImageId': [], 'EncodedPixels': []}

    for img_id in tqdm(test_names):
        imgs = []
        img_path = os.path.join(TEST_IMG_DIR, img_id)  # get certain image path
        imgs.append(imread(img_path) / 255.0)   # load image and normalize it

        y_pred = model(np.array(imgs))  # get prediction for an image

        # map the last dim from size 2 to 1 transforming probabilities into class labels
        mask = tf.argmax(y_pred, axis=-1)
        mask = mask[..., tf.newaxis].numpy()

        enc_mask = rle_encode(mask) # encode obtained mask with rle

        submission['ImageId'].append(img_id)
        submission['EncodedPixels'].append(enc_mask if len(enc_mask) > 0 else None)

    df = pd.DataFrame(submission).set_index(keys='ImageId', drop=True)
    df.to_csv(SAVE_PATH)


if __name__ == '__main__':
    TEST_IMG_DIR = '../data/test_v2/'  # test images directory
    LOAD_MODEL = '../model/final_model.h5'  # model to make predictions
    SAVE_PATH = '../submission/submission.csv'  # path to save predictions

    write_result()

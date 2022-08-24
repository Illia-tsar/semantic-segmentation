TRAIN_IMG_DIR = '../data/train_v2/'    # training images path
TEST_IMG_DIR = '../data/test_v2/'  # test images directory
SEGMENTATION = '../data/train_ship_segmentations_v2.csv'    # training images segmentations path
SAVE_MODEL = '../model/model3_weights.hdf5'  # model saving path
LOAD_MODEL = '../model/model3_weights.hdf5'    # path for model loading
SAVE_HISTORY = '../model/history/history3.pickle'   # history saving path
PLOT_MODEL_PATH = '../model/model.png'  # path specifies where to save generated model plot
SAVE_SUBMISSION = '../submission/submission.csv'  # path to save predictions
# plot history for training model on 192x192 images for 10 epochs and 384x384 images for 5 epochs
HISTORY_TO_PLOT = ['../model/history/history1.pickle', '../model/history/history2.pickle']
EPOCHS = 2
BATCH_SIZE = 4
MAX_STEPS_TRAIN = 1500
MAX_STEPS_VAL = 1500

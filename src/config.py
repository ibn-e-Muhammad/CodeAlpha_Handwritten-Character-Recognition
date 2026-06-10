import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'pipeline_vision', 'data')

# Data Files
TRAIN_CSV = os.path.join(DATA_DIR, 'emnist-balanced-train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'emnist-balanced-test.csv')
MAPPING_TXT = os.path.join(DATA_DIR, 'emnist-balanced-mapping.txt')

# Image Dimension Parameters
IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1

# Hyperparameters
NUM_CLASSES = 47
BATCH_SIZE = 64
EPOCHS = 15

# Export Paths
X_TRAIN_NPY = os.path.join(DATA_DIR, 'x_train.npy')
Y_TRAIN_NPY = os.path.join(DATA_DIR, 'y_train.npy')
X_TEST_NPY = os.path.join(DATA_DIR, 'x_test.npy')
Y_TEST_NPY = os.path.join(DATA_DIR, 'y_test.npy')
MAPPING_JSON = os.path.join(DATA_DIR, 'mapping.json')

from keras.applications import ResNet152V2, ResNet50V2
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.losses import Huber
import pandas as pd


# Load the best model
model = load_model('fried_noodles_best3.h5')
model.summary()

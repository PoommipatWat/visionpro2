import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

BATCH_SIZE = 20
IMAGE_SIZE = (300,200)

# Load data from the CSV file
dataframe = pd.read_csv('C:\\Users\\Poommipat\\Desktop\\ComVision\\shuffled_fried_noodles_dataset_normalized.csv')

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.9,1.1],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Create data generators for training, validation, and testing
train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[:1599],
    directory='C:\\Users\\Poommipat\\Desktop\\ComVision\\images',
    x_col='filename',
    y_col=['norm_meat', 'norm_veggie', 'norm_noodle'],
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

validation_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[1600:1699],
    directory='C:\\Users\\Poommipat\\Desktop\\ComVision\\images',
    x_col='filename',
    y_col=['norm_meat', 'norm_veggie', 'norm_noodle'],
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

test_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[1700:],
    directory='C:\\Users\\Poommipat\\Desktop\\ComVision\\images',
    x_col='filename',
    y_col=['norm_meat', 'norm_veggie', 'norm_noodle'],
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

# Define the CNN model
inputIm = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3,))
conv1 = Conv2D(256, kernel_size=(3, 3), activation='relu')(inputIm)
pool1 = MaxPool2D(pool_size=(3, 3))(conv1)
conv2 = Conv2D(512, kernel_size=(3, 3), activation='relu')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(1024, kernel_size=(3, 3), activation='relu')(pool2)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(2048, kernel_size=(3, 3), activation='relu')(pool3)
pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
flatten = Flatten()(pool4)
dense1 = Dense(1024, activation='relu')(flatten)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)
dense3 = Dense(256, activation='relu')(dropout1)
dropout3 = Dropout(0.5)(dense3)
predictedW = Dense(3, activation='sigmoid')(dropout3)

# Create the model
model = Model(inputs=inputIm, outputs=predictedW)

# Compile the model
model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mean_absolute_error'])

# Callbacks
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('mean_absolute_error'))
        self.val_losses.append(logs.get('val_mean_absolute_error'))

        plt.clf()
        plt.plot(self.x, self.losses, label='mean_absolute_error')
        plt.plot(self.x, self.val_losses, label='val_mean_absolute_error')
        plt.legend()
        plt.pause(0.01)

# ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint('C:\\Users\\Poommipat\\Desktop\\ComVision\\contest_final2.h5', verbose=1, monitor='val_mean_absolute_error', save_best_only=True, mode='min')
plot_losses = PlotLosses()

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, plot_losses])

# Evaluate the model on the test set
model = load_model('C:\\Users\\Poommipat\\Desktop\\ComVision\\contest_final2.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('Score (MSE, MAE):', score)

# Make predictions
test_generator.reset()
predictions = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers=1,
    use_multiprocessing=False)
print('Predictions:', predictions)
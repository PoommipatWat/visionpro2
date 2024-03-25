from keras.applications import NASNetLarge
from keras.models import Model, load_model
from keras.layers import Dense, Dropout,  GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.losses import Huber
import pandas as pd
from keras.regularizers import l2

BATCH_SIZE = 128
IMAGE_SIZE = (331, 331)

# Load dataset
dataframe = pd.read_csv('fried_noodles_dataset.csv', delimiter=',', header=0)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train generator
train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:1485],
    directory='images',
    x_col='filename',
    shuffle=True,
    y_col=['noodle'],
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other'
)

# Validation generator
validation_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[1486:1671],
    directory='images',
    x_col='filename',
    y_col=['noodle'],
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other'
)

# Test generator
test_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[1672:1855],
    directory='images',
    x_col='filename',
    y_col=['noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other'
)


base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

#fine tuning
base_model.trainable = False
for layer in base_model.layers[:-100]:
    layer.trainable = False

avarage_pooling_layer = GlobalAveragePooling2D()(base_model.output)
prediction_layer = Dense(2048, activation='relu')(avarage_pooling_layer)
prediction_layer = Dropout(0.5)(prediction_layer)
prediction_layer = Dense(1024, activation='relu')(avarage_pooling_layer)
prediction_layer = Dropout(0.5)(prediction_layer)
prediction_layer = Dense(512, activation='relu')(avarage_pooling_layer)
prediction_layer = Dropout(0.5)(prediction_layer)
prediction_layer = Dense(1, activation='relu')(prediction_layer)

model = Model(inputs=base_model.input, outputs=prediction_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(), metrics=['mae'])

checkpoint = ModelCheckpoint('model_best.h5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # Metric to be monitored
    factor=0.05,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=15,         # Number of epochs with no improvement after which learning rate will be reduced.
    verbose=1,          # If > 0, prints a message for each update.
    mode='auto',        # In 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.
    min_delta=0.0001,   # Threshold for measuring the new optimum, to only focus on significant changes.
    cooldown=0,         # Number of epochs to wait before resuming normal operation after lr has been reduced.
    min_lr=0.0000001      # Lower bound on the learning rate.
)

model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=10000,
                    callbacks=[
                        ModelCheckpoint('model.h5', save_best_only=True),
                        ReduceLROnPlateau(patience=2),
                        checkpoint,
                        reduce_lr
                    ])


# Load the best model
model = load_model('model.h5')

# Test the model
test_results = model.predict_generator(
    test_generator,
    steps=len(test_generator)
)

# Calculate MAE for each output
test_mae = []
for i in range(3):
    mae = abs(test_results[:, i] - test_generator.labels[:, i]).mean()
    test_mae.append(mae)

print('MAE for each output:')
res = ['noodle']
for i, mae in enumerate(test_mae):
    print(f'Output {res[i]}: {mae}')


# Test the model
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator)
)
print('score (mse, mae):', score)

test_generator.reset()
predictions = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers=1,
    use_multiprocessing=False
)
print('predictions:', predictions)



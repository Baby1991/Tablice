import tensorflow as tf
from tensorflow.keras import datasets, layers, models


model = models.Sequential()

model.add(layers.Conv2D(2, (3, 3), activation='relu',strides=(3,3), input_shape=(180, 120, 3), padding="same"))
model.add(layers.Conv2D(12, (3, 3), activation='relu',strides=(3,3), input_shape=(60, 40, 2), padding="same"))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(20, 14, 12), padding="same"))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(20, 14, 16), padding="same"))
model.add(layers.Conv2D(24, (3, 3), activation='relu', input_shape=(20, 14, 16), padding="same"))
model.add(layers.MaxPool2D((2,2), padding="same"))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(10, 7, 24), padding="same"))
model.add(layers.Conv2D(2, (3, 3), activation='relu', input_shape=(10, 7, 16), padding="same"))
model.add(layers.MaxPool2D((2,2), padding="same"))
model.add(layers.Flatten())
model.add(layers.Dense(32))
model.add(layers.Dense(1,activation='linear'))


model.summary()
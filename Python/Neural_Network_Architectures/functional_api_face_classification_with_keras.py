import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.utils as utils

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import os
import datetime


#%% DATA PREPROCESSION
data_path = "face_classification.csv"
data = pd.read_csv(data_path)

# convert pixels to numpy array
data["pixels"] = data["pixels"].apply(lambda x: np.array(x.split(), dtype='float32'))
# normalize pixel data 
data["pixels"] = data["pixels"].apply(lambda x: x / 255.)


#%% DATA EXPLORATION
print(data.info())
print(data.describe())

print(data.columns)
print("rows: ", len(data),
      "cols: ", len(data.columns))

unique_classes = (117, 5, 2)
print(unique_classes)

# create arrays
img_shapes = (len(data), 48, 48, 1)
x = np.array(data["pixels"].tolist()).reshape(*img_shapes)
y = np.array(data[["age", "ethnicity", "gender"]], dtype="int32")
# y = [data["age"], data["ethnicity"], data["gender"]]

# show images
rdm_idx = np.random.randint(0, len(data), size=9)
fig, axs = plt.subplots(3, 3, dpi=100, tight_layout=True)
for i, ax in zip(rdm_idx, axs.flatten()):
    ax.imshow(x[i], cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()


#%% CREATE DATASETS
(x_train, x_test,
 age_train, age_test,
 eth_train, eth_test,
 gender_train, gender_test) = train_test_split(x,
                                               y[:, 0], y[:, 1], y[:, 2],
                                               test_size=0.3,
                                               random_state=42)

batch_size = 32
shuffle_buffer_size = 100
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,
                                                    (age_train, eth_train, gender_train)))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,
                                                   (age_test, eth_test, gender_test))) 

train_dataloader = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
test_dataloader = test_dataset.batch(batch_size)


#%% CREATE MODEL
inputs = keras.Input(shape=img_shapes[1:])

x = L.Conv2D(filters=32, kernel_size=(3,3),
             activation="relu", name="first_conv2d")(inputs)
x = L.MaxPooling2D(pool_size=(2,2), name="first_maxpooling")(x)
first_output = L.Flatten(name="first_flatten")(x)

x = L.Conv2D(filters=32, kernel_size=(3,3),
             activation="relu", name="second_conv2d")(inputs)
x = L.MaxPooling2D(pool_size=(2,2), name="second_maxpooling")(x)
second_output = L.Flatten(name="second_flatten")(x)

x = L.Conv2D(filters=32, kernel_size=(3,3),
             activation="relu", name="third_conv2d")(inputs)
x = L.MaxPooling2D(pool_size=(2,2), name="third_maxpooling")(x)
third_output = L.Flatten(name="third_flatten")(x)

dense_input = L.add([first_output, second_output, third_output],
                    name="dense_input")

x = L.Dropout(0.5, name="dropout_1")(dense_input)
dense_output = L.Dense(128, activation="relu", name="dense_1")(x)

age_dropout = L.Dropout(0.25, name="age_dropout")(dense_output)
age_pred = L.Dense(unique_classes[0], activation="softmax",
                   name="age")(age_dropout)

ethnicity_dropout = L.Dropout(0.25, name="ethnicity_dropout")(dense_output)
ethnicity_pred = L.Dense(unique_classes[1], activation="softmax",
                         name="ethnicity")(ethnicity_dropout)

gender_dropout = L.Dropout(0.25, name="gender_dropout")(dense_output)
gender_pred = L.Dense(unique_classes[2], activation="softmax",
                      name="gender")(gender_dropout)

model = keras.Model(inputs=inputs,
                    outputs=[age_pred, ethnicity_pred, gender_pred])

utils.plot_model(model, "face_classification_model.png",
                 show_shapes=True)
print(model.summary())


#%% COMPILE THE MODEL
optimizer = keras.optimizers.Adam()

criterion = {"age": keras.losses.SparseCategoricalCrossentropy(),
             "ethnicity": keras.losses.SparseCategoricalCrossentropy(),
             "gender": keras.losses.SparseCategoricalCrossentropy()}

metrics = {"age": keras.metrics.SparseCategoricalAccuracy(),
           "ethnicity": keras.metrics.SparseCategoricalAccuracy(),
           "gender": keras.metrics.SparseCategoricalAccuracy()}

model.compile(optimizer=optimizer,
              loss=criterion,
              metrics=metrics)


#%% TRAINING THE MODEL
logdir = os.path.join("logs", "face_classification", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=10)

tensorboard_callback = keras.callbacks.TensorBoard(logdir,
                                             histogram_freq=1,
                                             write_graph=True,
                                             write_images=True)

epochs = 100
history = model.fit(train_dataloader,
                    validation_data=test_dataloader,
                    epochs=epochs,
                    callbacks=[early_stopping, tensorboard_callback])

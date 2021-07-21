# -*- coding: utf-8 -*-

import os
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


base_dir = os.getcwd() + "\\yoga pose"

train_dir = os.path.join(base_dir, "train")

validation_dir = os.path.join(base_dir, "val")

test_dir = os.path.join(base_dir, "test")


train_downdog_dir = os.path.join(train_dir, "downdog")
train_goddess_dir = os.path.join(train_dir, "goddess")
train_plank_dir = os.path.join(train_dir, "plank")
train_tree_dir = os.path.join(train_dir, "tree")
train_warrior2_dir = os.path.join(train_dir, "warrior2")


validation_downdog_dir = os.path.join(validation_dir, "downdog")
validation_goddess_dir = os.path.join(validation_dir, "goddess")
validation_plank_dir = os.path.join(validation_dir, "plank")
validation_tree_dir = os.path.join(validation_dir, "tree")
validation_warrior2_dir = os.path.join(validation_dir, "warrior2")


test_downdog_dir = os.path.join(test_dir, "downdog")
test_goddess_dir = os.path.join(test_dir, "goddess")
test_plank_dir = os.path.join(test_dir, "plank")
test_tree_dir = os.path.join(test_dir, "tree")
test_warrior2_dir = os.path.join(test_dir, "warrior2")


print("total training downdog pose images:", len(os.listdir(train_downdog_dir)))
print("total training goddess pose images:", len(os.listdir(train_goddess_dir)))
print("total training plank pose images:", len(os.listdir(train_plank_dir)))
print("total training tree pose images:", len(os.listdir(train_tree_dir)))
print("total training warrior2 pose images:", len(os.listdir(train_warrior2_dir)))

print("total validation downdog pose images:", len(os.listdir(validation_downdog_dir)))
print("total validation goddess pose images:", len(os.listdir(validation_goddess_dir)))
print("total validation plank pose images:", len(os.listdir(validation_plank_dir)))
print("total validation tree pose images:", len(os.listdir(validation_tree_dir)))
print(
    "total validation warrior2 pose images:", len(os.listdir(validation_warrior2_dir))
)

print("total test downdog pose images:", len(os.listdir(test_downdog_dir)))
print("total test goddess pose images:", len(os.listdir(test_goddess_dir)))
print("total test plank pose images:", len(os.listdir(test_plank_dir)))
print("total test tree pose images:", len(os.listdir(test_tree_dir)))
print("total test warrior2 pose images:", len(os.listdir(test_warrior2_dir)))


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(5, activation="softmax"))
model.summary()


model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["acc"],
)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=25,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)


history = model.fit(
    train_generator,
    steps_per_epoch=30,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=10,
)


for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))


plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()


for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

model.save("yogaPoseClassifier.model")


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(5, activation="softmax"))


model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["acc"],
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=25,
    class_mode="categorical",
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)


plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Test accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Test loss")
plt.title("Training and Test loss")
plt.legend()
plt.show()

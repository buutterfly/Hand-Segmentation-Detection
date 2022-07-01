import cv2
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

data = np.load('data.npy')
label = np.load('label.npy')

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)


def preprocessing(img_tr):
    img_tr = cv2.cvtColor(img_tr, cv2.COLOR_BGR2GRAY)
    normalized = img_tr / 255.0
    return normalized


x_train = np.array(list(map(preprocessing, x_train)))
x_test = np.array(list(map(preprocessing, x_test)))
x_validation = np.array(list(map(preprocessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10)

dataGen.fit(x_train)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
y_validation = to_categorical(y_validation, 2)

model = Sequential()
model.add((Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu')))
model.add((Conv2D(32, (3, 3), activation='relu')))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add((Conv2D(64, (3, 3), activation='relu')))
model.add((Conv2D(64, (3, 3), activation='relu')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

steps_per_epoch_val = len(x_train) // 64

history = model.fit(dataGen.flow(x_train, y_train, batch_size=64),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=10,
                    validation_data=(x_validation, y_validation),
                    shuffle=1)

print(model.evaluate(x_test, y_test))

model.save("HandTrainingModel.h5")

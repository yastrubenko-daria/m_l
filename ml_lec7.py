# Нейроный сети:
# - сверточные (конволюционные) нейроные сети (CNN) -копьютерное зрение,классификация изображений
# - рекуррентные нейроные сети (RNN) - распознование рукописны=ого текста, обработка естественного языка
# - генеративные состязательные сети (GAN) - создание художественных, музыкальных произведений
# - многослойный перцептрон - простейший тип НС

# Данная сеть способна сложить 2+2 или другие небольшие значения
# веса на связь
# w0 = 0.9907079
# w1 = 1.0264927
# w2 = 0.01417504
# w3 = -0.8950311
# w4 = 0.88046944
# w5 = 0.7524377
# w6 = 0.794296
# w7 = 1.1687347
# w8 = 0.2406084
#
# # смещения
# b0 = -0.00070612
# b1 = -0.06846002
# b2 = -0.00055442
# b3 = -0.00000929
#
#
# def relu(x):
#     return max(0, x)
#
#
# def predict(x1, x2):
#     h1 = (x1 * w0) + (x2 * w1) + b0
#     h2 = (x1 * w2) + (x2 * w3) + b1
#     h3 = (x1 * w4) + (x2 * w5) + b2
#
#     y = (relu(h1) * w6) + (relu(h2) * w7) + (relu(h3) * w8) + b3
#     return y
#
#
# print(predict(2, 2))
#
# print(predict(1.5, 1.5))

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import numpy as np
#
# img_path = './cat.png'
# img = image.load_img(img_path, target_size = (224,224))
#
# img_array = image.img_to_array(img)
# print(img_array.shape)
#
# img_batch = np.expand_dims(img_array, axis=0)
# print(img_batch.shape)
# from tensorflow.keras.applications.resnet50 import preprocess_input
#
# img_processed = preprocess_input(img_batch)
#
# from tensorflow.keras.applications.resnet50 import ResNet50
#
# model = ResNet50()
# prediction = model.predict(img_processed)
# from tensorflow.keras.applications.resnet50 import decode_predictions
# print(decode_predictions(prediction, top=5)[0])
#

# plt.imshow(img)
# plt.show()

#cat/dogs

import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing import image

import numpy as np
import tensorflow


TRAIN_DATA_DIR = './data/test/train_test/'
VALIDATION_DATA_DIR = './data/test/val_test/'
TRAIN_SAMPLES = 500 #колчиство картинок
VALIDATION_DATA = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224,224
BATCH_SIZE = 64

#БИНАРНАЯ, определяем кошек и собак

#сгенерируем дополнительные данные
# noinspection PyUnresolvedReferences
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
# noinspection PyUnresolvedReferences
from tensorflow.keras.applications.resnet50 import ResNet50

train_datagen = image.ImageDataGenerator(
    preprocessind_function=preprocess_input,
    rotation_range=20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
)

val_datagen = image.ImageDataGenerator(
    preprocessind_function=preprocess_input,
    )

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH,IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed = 12345,
    class_mode='categorical',
)
val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH,IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical',
)
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout, GlobalAveragePooling20)
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model

def model_maker():
    base_model = MobileNet(include_top=False,
                           input_shape = (IMG_WIDTH,IMG_HEIGHT),
                           )
    for layer in base_model.layers[:]:
        layer.trainable =False

    input = Input(shape= (IMG_WIDTH, IMG_HEIGHT,3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling20()(custom_model)
    custom_model = Dense(64, activation = 'relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation ='softmax')(custom_model)
    return Model(inputs=input, outputs=prediction)

# noinspection PyUnresolvedReferences
from tensorflow.keras.optimizers import Adam

model = model_maker()
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)
import math
num_steps = math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10,#шаг обучения
    validation_data=val_generator,
    validation_steps=num_steps,
)

print(val_generator.class_indices)

model.save('./data/test/model.h5')

#testing

import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'


# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing import image

model = load_model('./data/test/model.h5')

img_path = './cat.png'
img = image.load_img(img_path, target_size = (224,224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# noinspection PyUnresolvedReferences
from tensorflow.keras.applications.mobilenet import preprocess_input

img_processed = preprocess_input(img_batch)

prediction = model.predict(img_processed)






import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.applications import InceptionV3

from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

#Создаем нашу сеть
inc_model=InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape=((150, 150, 3)))

bottleneck_datagen = ImageDataGenerator(rescale=1. / 255)  # собственно, генератор


train_generator = bottleneck_datagen.flow_from_directory('Train',
                                                         target_size=(150, 150),
                                                         batch_size=32,
                                                         class_mode=None,
                                                         shuffle=False)

validation_generator = bottleneck_datagen.flow_from_directory('Validation',
                                                               target_size=(150, 150),
                                                               batch_size=32,
                                                               class_mode=None,
                                                               shuffle=False)
print("begin")

bottleneck_features_train = inc_model.predict_generator(train_generator, 40)
np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

print("second stage")

bottleneck_features_validation = inc_model.predict_generator(validation_generator, 38)
np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)

print("done")

# train_data = np.load(open('bottleneck_features_and_weights/bn_features_train.npy', 'rb'))
# train_labels = np.array([0] * 373 + [1] * 967)
#
# validation_data = np.load(open('bottleneck_features_and_weights/bn_features_validation.npy', 'rb'))
# validation_labels = np.array([0] * 19 + [1] * 19)
#
# fc_model = Sequential()
# fc_model.add(Flatten(input_shape=train_data.shape[1:]))
# fc_model.add(Dense(64, activation='relu', name='dense_one'))
# fc_model.add(Dropout(0.5, name='dropout_one'))
# fc_model.add(Dense(64, activation='relu', name='dense_two'))
# fc_model.add(Dropout(0.5, name='dropout_two'))
# fc_model.add(Dense(1, activation='sigmoid', name='output'))
#
# fc_model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# fc_model.fit(train_data, train_labels,
#             nb_epoch=50, batch_size=16,
#             validation_data=(validation_data, validation_labels))
#
# fc_model.save_weights('bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5') # сохраняем веса\
# fc_model.evaluate(validation_data, validation_labels)
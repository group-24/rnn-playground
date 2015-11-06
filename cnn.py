import keras
from keras.datasets import cifar10
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

nb_classes = 11
# convert class vectors to binary class matrices
# converts a number to unary so 4 is 0001
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
#p = plt.imshow(X_train[1].T)
#p.show()
nb_filters = 32
nb_conv = 3 
nb_pool = 2
model = Sequential()
model.add(Convolution2D(nb_filters, 3,3, input_shape=(3,32,32)))
model.add(Activation('sigmoid'))
model.add(Convolution2D(nb_filters, 3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=1000, nb_epoch=5, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

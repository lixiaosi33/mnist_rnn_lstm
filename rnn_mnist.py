"""
Implements a Recurrent Neural Netweork with the
recurrent weight matrix intialized by identity matrix

Optimizer used: RMSprop
Loss function: categorical_crossentropy
Reference: http://arxiv.org/pdf/1504.00941v2.pdf
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from keras.initializations import identity, normal 
from keras.utils import np_utils
import os
import sys
import cPickle as pickle
import time


if len(sys.argv)>1:
    fname = sys.argv[1]
else:
    print ("Requires a file to store the output results")
    exit(0)

# to record the execution time
start_time = time.clock()

# architecture details
output_classes = 10
hidden_units = 100

# learning rate parameter
learning_rate = 1e-6

# running details
num_epochs = 2
batch_size = 32


# load the mnist data and split it between train and test sets
# train set: 60,000 examples (28 x 28 images)
# test set: 10,000 examples (28 x 28 images)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the examples to a vector of length 28 x 28 = 784
X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)

# Convert the data type to float before normalizing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the pixel values such 
# that they lie between 0 to 1
X_train /= 255
X_test /= 255

# convert class labels to binary class vectors 
# eg., 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Y_train = np_utils.to_categorical(y_train, output_classes)
Y_test = np_utils.to_categorical(y_test, output_classes)


# Build the sequential model
model = Sequential()
model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=X_train.shape[1:]))
model.add(Dense(output_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

print('RNN Model Evaluation:')

# Train the model for a fixed number of epochs
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs, verbose=1, 
            validation_data=(X_test, Y_test))

# Compute the loss on the input data, batch by batch
scores = model.evaluate(X_test, Y_test, verbose=0)

test_score = scores[0]
test_accuracy = scores[1]
print('RNN Model Test score:', test_score)
print('RNN Model Test accuracy:', test_accuracy)

execution_time = time.clock() - start_time

data = {
        'test_score': test_score,
        'test_accuracy': test_accuracy,
        'hist': hist, 
        'execution_time': execution_time,
        }

pickle.dump(data, open(fname, 'wb'))
print ("pickle complete")
print (fname)
print ("Execution time: ", execution_time)


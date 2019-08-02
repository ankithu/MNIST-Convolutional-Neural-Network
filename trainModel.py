import pickle
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils
import time

NAME = "mnist-cnn-dense{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in) #training data
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in) # labels

X = X/255.0
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
#model = Sequential()
#model.add(Conv2D(64, (3,3),	 input_shape = X.shape[1:]))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size = (2,2)))

#model.add(Conv2D(64, (3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size = (2,2)))

#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation("relu"))

#model.add(Dense(10))
#model.add(Activation('sigmoid'))

model.compile(loss="categorical_crossentropy",
			  optimizer="rmsprop",
			  metrics=['accuracy'])

#one_hot_labels = utils.to_categorical(y, num_classes=10)
#print(one_hot_labels)
model.fit(X,y,batch_size=32, epochs = 3, validation_split=0.1, callbacks=[tensorboard])
model.save('mnist-cnn-dense')

import numpy as np 
import os
import cv2
import random
import pickle
import tensorflow
from tensorflow.keras import utils

DATADIR = "/Users/ankithudupa/Documents/Personal Projects/Python/MNIST CNN/trainingSet"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMG_SIZE = 28

training_data = []

def create_training_data():
	curHot = 0
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category) # navigate into dir
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([img_array, class_num])
			except Exception as e:
				pass

create_training_data()
print(len(training_data))
random.shuffle(training_data)

X = [] #features
y = [] #labels
for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
one_hot_labels = utils.to_categorical(y, num_classes=10)
print(one_hot_labels)
print(X[1])
print(one_hot_labels[1])
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(one_hot_labels, pickle_out)
pickle_out.close()
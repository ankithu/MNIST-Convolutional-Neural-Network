import cv2 
import tensorflow as tf 

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def prepare(filepath):
	IMG_SIZE = 28
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("mnist-cnn-dense")

prediction = model.predict([prepare('trainingSet/5/img_1001.jpg')])
print(prediction)
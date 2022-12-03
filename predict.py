import time
from keras.models import model_from_json
import cv2
import numpy as np

# Loading the model
json_file = open("melanoma.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("melanoma.h5")
print("Loaded model from disk")

image_non_melanoma = cv2.imread('DermMel/valid/NotMelanoma/ISIC_0024318.jpg')

image_melanoma = cv2.imread('DermMel/valid/Melanoma/AUG_0_25.jpeg')

img = cv2.resize(image_non_melanoma, (224, 224))
img = np.reshape(img, [1, 224, 224, 3])

img1 = cv2.resize(image_melanoma, (224, 224))
img1 = np.reshape(img1, [1, 224, 224, 3])

predictionsold = loaded_model.predict(img)
print('Model')
# print(classes)
classes = np.argmax(predictionsold, axis=1)
print(classes)

output = {0: 'melanoma', 1: 'non melanoma'}

classes = np.argmax(predictionsold, axis=1)

print('Predicted Value -> ', output[int(classes)])
print('****************************************************')

cv2.imshow('Not Melanoma', image_non_melanoma)
cv2.imshow('Melanoma', image_melanoma)
cv2.waitKey(0)
cv2.destroyAllWindows(10)

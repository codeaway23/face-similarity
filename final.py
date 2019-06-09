import cv2
import numpy as np
from keras.models import load_model

#image paths
img1_path = 'data/Zico/Zico_0002.jpg'
img2_path = 'data/Zico/Zico_0003.jpg'

#image arrays
img_a = cv2.imread(img1_path).reshape(1,250,250,3)
img_b = cv2.imread(img2_path).reshape(1,250,250,3)

#class prediction using our trained model
model = load_model('model.h5')
pred = model.predict({'image1': img_a, 'image2': img_b})
if pred < 0.5:
	class_ = 'No'
else:
	class_ = 'Yes'

print("Image similarity metric as per our trained model: ", pred[0][0])
print("Prediction - Are both images of the same person?", class_)


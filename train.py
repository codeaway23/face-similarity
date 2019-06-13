from utils import *

import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

# image augmentation hurts the training. hence commented out.
datagen = ImageDataGenerator(rescale=1./255.) 
			     # rotation_range=20.,
			     # width_shift_range=0.2,
			     # height_shift_range=0.2,
			     # horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255.)

h = 250 #input image height
w = 250 #input image width
batch = 32 #batch size
inp_shape = (h, w, 3) #input image shape
train_num = 4000 #number of training samples
test_num = 500 #number of testing samples
nb_epochs =5 #number of epochs

# preprocessing
true_pairs, false_pairs = get_pairs()
df_train, df_test = prepare_data(true_pairs, false_pairs, train_num, test_num)
inp_gen = multi_input_generator(df_train, datagen, batch, h, w)
test_gen = multi_input_generator(df_test, test_datagen, batch, h, w)

#build model
model = build_model(inp_shape)
print(model.summary())

#compile and train model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
hist = model.fit_generator(inp_gen,
		           epochs = nb_epochs,
		           steps_per_epoch=train_num/batch,
		           validation_data = test_gen,
		           validation_steps=test_num/batch,
		           shuffle=False)

#get validation accuracy
print("validation accuracy:", hist.history['val_acc'][-1])

#save model
model.save('model.h5')

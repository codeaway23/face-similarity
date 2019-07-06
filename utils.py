import os
import cv2
import random
import numpy as np 
import pandas as pd
from random import shuffle
from itertools import combinations

from keras.models import Model
from keras.engine.input_layer import Input
from keras.applications import inception_v3
from keras.layers import Concatenate, Dense, Flatten

from keras_preprocessing.image import ImageDataGenerator

# random.seed(42)

#split images into pairs of same people and different people
def get_pairs():

	BASE_DIR = 'data'
	names = [BASE_DIR + '/' + x for x in os.listdir(BASE_DIR)]

	print("getting pairs of same faces...")
	true_pairs= [list(combinations([x + '/' + y for y in os.listdir(x)], r=2)) for x in names]
	true_pairs = [x for x in true_pairs if x!=[]]
	true_pairs = [y for x in true_pairs for y in x]

	count = [len(os.listdir(x)) for x in names]
	idx_ones = [i for i,_ in enumerate(count) if count[i]==1]

	# majority of possibilities are omitted here since there are far lesser 
	# true pairs as compared to possible false pairs. most combinations falling
	# in the latter category have been omitted to maintain class balance in the
	# dataset

	print("getting pairs of different faces...")
	idx_multiple = [i for i,_ in enumerate(count) if count[i]>1]
	false_pairs = list(combinations([names[x] + '/' + os.listdir(names[x])[0] for x in idx_multiple], r=2))

	return true_pairs, false_pairs

def prepare_data(true_pairs, false_pairs, train_num, test_num):
	
	total = train_num + test_num 

	shuffle(true_pairs)
	shuffle(false_pairs)

	def get_dataframe(img1, img2, labels):

		df = pd.DataFrame({'image1': img1,
				   'image2': img2,
				   'labels': labels})

		return df

	# A 50-50 ratio is maintained for both classes
	img1_true = [x[0] for x in true_pairs[:int(train_num/2)]]
	img2_true = [x[1] for x in true_pairs[:int(train_num/2)]]
	lab1 = ['yes']*len(img1_true)

	img1_false = [x[0] for x in false_pairs[:int(train_num/2)]]
	img2_false = [x[1] for x in false_pairs[:int(train_num/2)]]
	lab2 = ['no']*len(img1_false)

	img1 = img1_true + img1_false
	img2 = img2_true + img2_false
	labels = list(lab1) + list(lab2)

	img1_true = [x[0] for x in true_pairs[int(train_num/2):int(total/2)]]
	img2_true = [x[1] for x in true_pairs[int(train_num/2):int(total/2)]]
	lab1 = ['yes']*len(img1_true)

	img1_false = [x[0] for x in false_pairs[int(train_num/2):int(total/2)]]
	img2_false = [x[1] for x in false_pairs[int(train_num/2):int(total/2)]]
	lab2 = ['no']*len(img1_false)

	# data shuffled and put in a dataframe for image data generators
	shuffled = list(zip(img1, img2, labels))
	shuffle(shuffled)
	img1, img2, labels = zip(*shuffled)

	df_train = get_dataframe(img1, img2, labels)

	test_img1 = img1_true + img1_false
	test_img2 = img2_true + img2_false
	test_labels = list(lab1) + list(lab2)

	shuffled = list(zip(test_img1, test_img2, test_labels))
	shuffle(shuffled)
	test_img1, test_img2, test_labels = zip(*shuffled)

	df_test = get_dataframe(test_img1, test_img2, test_labels)

	return df_train, df_test

def build_model(inp_shape):

	# inception v3 pre-trained iamgenet weights used
	conv = inception_v3.InceptionV3(input_shape=inp_shape, include_top=False, weights='imagenet')
	# inception v3 parameters are frozen
	for layer in conv.layers:
		conv.trainable = False

	inp1 = Input(shape=inp_shape, name='image1')
	inp2 = Input(shape=inp_shape, name='image2')
	conv1 = conv(inp1)
	flat1 = Flatten()(conv1)
	conv2 = conv(inp2)
	flat2 = Flatten()(conv2)
	conc = Concatenate(axis=1)([flat1, flat2])
	dense = Dense(16, activation='relu', name='dense')(conc)
	out = Dense(1, activation='sigmoid', name='output')(dense)

	model = Model(inputs=[inp1, inp2], outputs=[out])

	return model

def multi_input_generator(df, generator, batch_size, img_height, img_width):

    genX1 = generator.flow_from_dataframe(dataframe=df,
  					  directory=None,
    					  x_col='image1',
    					  y_col='labels',
                                          target_size = (img_height,img_width),
                                          class_mode = 'binary',
                                          classes = ['yes', 'no'],
                                          batch_size = batch_size,
                                          shuffle=False)
    
    genX2 = generator.flow_from_dataframe(dataframe=df,
    					  directory=None,
    					  x_col='image2',
    					  y_col='labels',
                                          target_size = (img_height,img_width),
                                          class_mode = 'binary',
                                          classes = ['yes', 'no'],
                                          batch_size = batch_size,
                                          shuffle=False)

    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  


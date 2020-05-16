#! -*- coding: utf-8 -*-
# the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112

import numpy as np
from Capsule_Keras import *
from keras import utils
from keras.models import Model
from keras.layers import *
from keras import backend as K
from sklearn.model_selection import train_test_split
import sys
import argparse


# configuration
parser = argparse.ArgumentParser(description='scCapsNet')
# system config
parser.add_argument('--inputdata', type=str, default='data/PBMC_data.npy', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='data/PBMC_celltype.npy', help='address for celltype label')
parser.add_argument('--num_classes', type=int, default=8, help='number of cell type')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')
args = parser.parse_args()


inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms

data = np.load(inputdata)
labels = np.load(inputcelltype)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

z_dim = 16

input_size = x_train.shape[1]
x_in = Input(shape=(input_size,))
x = x_in
x1 = Dense(z_dim, activation='relu')(x_in)
x2 = Dense(z_dim, activation='relu')(x_in)
x3 = Dense(z_dim, activation='relu')(x_in)
x4 = Dense(z_dim, activation='relu')(x_in)
x5 = Dense(z_dim, activation='relu')(x_in)
x6 = Dense(z_dim, activation='relu')(x_in)
x7 = Dense(z_dim, activation='relu')(x_in)
x8 = Dense(z_dim, activation='relu')(x_in)
x9 = Dense(z_dim, activation='relu')(x_in)
x10 = Dense(z_dim, activation='relu')(x_in)
x11 = Dense(z_dim, activation='relu')(x_in)
x12 = Dense(z_dim, activation='relu')(x_in)
x13 = Dense(z_dim, activation='relu')(x_in)
x14 = Dense(z_dim, activation='relu')(x_in)
x15 = Dense(z_dim, activation='relu')(x_in)
x16 = Dense(z_dim, activation='relu')(x_in)

x = Concatenate()([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16])
x = Reshape((16, z_dim))(x)
capsule = Capsule(num_classes, 16, 3, False)(x) 
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule)

model = Model(inputs=x_in, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=400,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

model.save_weights('Modelweight.weight')

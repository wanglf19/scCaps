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

parser = argparse.ArgumentParser(description='scCapsNet')
# system config
parser.add_argument('--inputdata', type=str, default='data/PBMC_data.npy', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='data/PBMC_celltype.npy', help='address for celltype label')
parser.add_argument('--num_classes', type=int, default=8, help='number of cell type')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')
parser.add_argument('--dim_capsule', type=int, default=16, help='dimension of the capsule')
parser.add_argument('--num_capsule', type=int, default=16, help='number of the primary capsule')
parser.add_argument('--batch_size', type=int, default=400, help='training parameters_batch_size')
parser.add_argument('--epochs', type=int, default=10, help='training parameters_epochs')
parser.add_argument('--Model_weights', type=str, default='Modelweight.weights', help='Model_weight')


args = parser.parse_args()

inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms
z_dim = args.dim_capsule
num_capsule = args.num_capsule
epochs = args.epochs
batch_size = args.batch_size
Model_weights = args.Model_weights

data = np.load(inputdata)
labels = np.load(inputcelltype)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


input_size = x_train.shape[1]
print(input_size)

x_in = Input(shape=(input_size,))
x = x_in
x_all = list(np.zeros((num_capsule,1)))
encoders = []
for i in range(num_capsule):
    x_all[i] = Dense(z_dim, activation='relu')(x_in)
    encoders.append(Model(x_in, x_all[i]))

x = Concatenate()(x_all)
x = Reshape((num_capsule, z_dim))(x)
capsule = Capsule(num_classes, z_dim, 3, False)(x)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule)

model = Model(inputs=x_in, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save_weights(Model_weights)

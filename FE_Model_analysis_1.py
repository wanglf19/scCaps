#! -*- coding: utf-8 -*-
# the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112

from Visualization_Capsule_Keras import *
from keras import utils
from keras.models import Model
from keras.layers import *
from keras import backend as K
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='scCapsNet')
# system config
parser.add_argument('--inputdata', type=str, default='data/PBMC_data.npy', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='data/PBMC_celltype.npy', help='address for celltype label')
parser.add_argument('--num_classes', type=int, default=8, help='number of cell type')
parser.add_argument('--randoms', type=int, default=30, help='random number to split dataset')
parser.add_argument('--dim_capsule', type=int, default=16, help='dimension of the capsule')
parser.add_argument('--num_capsule', type=int, default=16, help='number of the primary capsule')
parser.add_argument('--weights', type=str, default='Modelweight.weights', help='trained weights')

args = parser.parse_args()
print("Loading...")
inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms
z_dim = args.dim_capsule
num_capsule = args.num_capsule

data = np.load(inputdata)
labels = np.load(inputcelltype)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


###################################################################################################
#1. model
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
output = capsule

model = Model(inputs=x_in, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])
#model.summary()
model.load_weights(args.weights)


###################################################################################################
#2 heatmap for coupling coefficients
Y_pred = model.predict(x_test)

coupling_coefficients_value = {}
count = {}
for i in range(len(Y_pred)):
    ind = int(Y_test[i])
    if ind in coupling_coefficients_value.keys():
        coupling_coefficients_value[ind] = coupling_coefficients_value[ind] + Y_pred[i]
        count[ind] = count[ind] + 1
    if ind not in coupling_coefficients_value.keys():
        coupling_coefficients_value[ind] = Y_pred[i]
        count[ind] = 1

total = np.zeros((num_classes,num_capsule))

plt.figure(figsize=(20,np.ceil(num_classes/4)*4))
for i in range(num_classes):
    average = coupling_coefficients_value[i]/count[i]
    Lindex = i + 1
    plt.subplot(np.ceil(num_classes/4),4,Lindex)
    total[i] = average[i]
    df = DataFrame(np.asmatrix(average))
    heatmap = sns.heatmap(df)
plt.savefig("FE_Model_analysis_1_heatmap.png")
plt.show()

###################################################################################################
#overall heatmap
plt.figure()
df = DataFrame(np.asmatrix(total))
heatmap = sns.heatmap(df)

plt.ylabel('Type capsule', fontsize=10)
plt.xlabel('Primary capsule', fontsize=10)
plt.savefig("FE_Model_analysis_1_overall_heatmap.png")
plt.show()

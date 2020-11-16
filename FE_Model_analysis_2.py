#! -*- coding: utf-8 -*-
# the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112
from Capsule_Keras import *
import numpy as np
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
parser.add_argument('--PC', type=int, default=1, help='indicate which principle component will be analyzed ')
parser.add_argument('--Primary_capsule', type=int, default=4, help='indicate which primary capsule will be analyzed')
parser.add_argument('--Cell_type', type=int, default=-1, help='indicate which cell type will be analyzed')

args = parser.parse_args()

print("Loading...")
inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms
z_dim = args.dim_capsule
num_capsule = args.num_capsule
PC = args.PC-1
Primary_capsule = args.Primary_capsule
Cell_type = args.Cell_type

data = np.load(inputdata)
labels = np.load(inputcelltype)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

###################################################################################################
#1. model
input_size = x_train.shape[1]

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

#model.summary()
model.load_weights(args.weights)

###################################################################################################
#2. select genes along principle component

#get all weight matrix
primary_capsule_encoder_weights = []
for i in range(num_capsule):
    primary_capsule_encoder_weights.append(encoders[i].get_weights()[0])

color = ['red','darksalmon','sienna','gold','olivedrab','darkgreen','chartreuse','darkcyan','deepskyblue','blue','darkorchid','plum','m','mediumvioletred','k','palevioletred']

#pick the weights of user specify Primary_capsule then PCA
totalweight = primary_capsule_encoder_weights[Primary_capsule]
pca = PCA(n_components=16)
pca.fit(totalweight)
weightpca = pca.transform(totalweight)

#Fragmentation to facilitate computation
ratio_plot = np.zeros((num_classes, 31))
difference = (np.max(weightpca[:, PC]) - np.min(weightpca[:, PC])) / 30
#print(np.max(weightpca[:, PC]), np.min(weightpca[:, PC]))


###################################################################################################
#2.1 select genes from maximum to minimum
print("select genes from maximum to minimum...")
#line plot
x_new_test = copy.deepcopy(x_test)
select_genes = []
dotted_line = -1

#calculate the prediction accuracy
plot_x = []
for j in range(31):
    #print(j)
    plot_x.append(np.max(weightpca[:, PC]) - difference * j)
    # sub A select genes
    gene_count = 0
    for i in range(input_size):
        if weightpca[i, PC] > (np.max(weightpca[:, PC]) - difference * j):
            gene_count = gene_count + 1
            rownum = x_new_test.shape[0]
            x_new_test[:, i] = np.zeros(rownum)
            #selected genes
            if dotted_line < 0 and i not in select_genes:
                select_genes.append(i)

    #print(gene_count)
    # sub B calculate the accuracy
    Y_pred = model.predict(x_new_test)
    Y_pred_order = np.argsort(Y_pred, axis=1)
    Y_pred_1 = Y_pred_order[:, num_classes-1]

    current_type = 0
    total = np.zeros((num_classes,1))
    correct = np.zeros((num_classes,1))

    for i in range(x_test.shape[0]):
        index_int = int(Y_test[i])
        if Y_test[i] == Y_pred_1[i]:
            correct[index_int] = correct[index_int] + 1
        total[index_int] = total[index_int] + 1

    ratio_drop = np.zeros((num_classes,1))
    for i in range(len(total)):
        ratio_drop[i] = correct[i] / total[i]

    for i in range(len(total)):
        ratio_plot[i, j] = ratio_drop[i]
        #find the position of dotted line
        if dotted_line<0 and Cell_type==i and ratio_plot[i, j] < 0.01 :
                dotted_line = j

#plot
plt.figure(figsize=(20,12))
ax = plt.subplot(2,2,1)
for i in range(num_classes):
    plt.plot(plot_x,ratio_plot[i], c=color[i], label = str(i))
if dotted_line > 0:
    dotted_line_pos = np.max(weightpca[:, PC]) - difference * dotted_line
    plt.plot([dotted_line_pos,dotted_line_pos],[1.0,0], 'k--',linewidth=3.0)

ax.invert_xaxis()
plt.legend(loc='lower left')
plt.xlabel('Masking genes along PC'+ str(PC+1))
plt.ylabel('Prediction accuracy(%)')
plt.title('Max2Min Primary Capsule'+ ' '+ str(Primary_capsule) +'-Type'+ str(Cell_type) )
#plt.show()

#scatter plot
plt.subplot(2,2,2)
plt.scatter(weightpca[:, 0], weightpca[:, 1],color='r', s=5,alpha=0.5,label = 'gene')
if dotted_line < 0:
    select_genes = []

np.save("Max2Min_genes.npy",np.asarray(select_genes))
plt.scatter(weightpca[select_genes, 0], weightpca[select_genes, 1], color = 'b',s=6,label = 'select_gene')
plt.legend(loc='lower left')

plt.ylabel('PC2', fontsize=10)
plt.xlabel('PC1', fontsize=10)
plt.title('Max2Min Primary Capsule'+ ' '+ str(Primary_capsule) +'-Type'+ str(Cell_type) )
#plt.show()


###################################################################################################
#2.2 select genes from minimum to maximum
print("select genes from minimum to maximum...")
#line plot
x_new_test = copy.deepcopy(x_test)
select_genes = []
dotted_line = -1

#calculate the prediction accuracy
plot_x = []
for j in range(31):
    #print(j)
    plot_x.append(np.min(weightpca[:, PC]) + difference * j)
    # sub A select genes
    gene_count = 0
    for i in range(input_size):
        if weightpca[i, PC] < (np.min(weightpca[:, PC]) + difference * j):
            gene_count = gene_count + 1
            rownum = x_new_test.shape[0]
            x_new_test[:, i] = np.zeros(rownum)
            # selected genes
            if dotted_line < 0 and i not in select_genes:
                select_genes.append(i)

    #print(gene_count)
    # sub B calculate the accuracy
    Y_pred = model.predict(x_new_test)
    Y_pred_order = np.argsort(Y_pred, axis=1)
    Y_pred_1 = Y_pred_order[:, num_classes-1]

    current_type = 0
    total = np.zeros((num_classes,1))
    correct = np.zeros((num_classes,1))

    for i in range(x_test.shape[0]):
        index_int = int(Y_test[i])
        if Y_test[i] == Y_pred_1[i]:
            correct[index_int] = correct[index_int] + 1
        total[index_int] = total[index_int] + 1

    ratio_drop = np.zeros((num_classes,1))
    for i in range(len(total)):
        ratio_drop[i] = correct[i] / total[i]

    for i in range(len(total)):
        ratio_plot[i, j] = ratio_drop[i]
        # find the position of dotted line
        if dotted_line < 0 and Cell_type == i and ratio_plot[i, j] < 0.01:
            dotted_line = j

#plot
ax= plt.subplot(2,2,3)
for i in range(num_classes):
    plt.plot(plot_x, ratio_plot[i], c=color[i], label = str(i))

if dotted_line > 0:
    dotted_line_pos = np.min(weightpca[:, PC]) + difference * dotted_line
    plt.plot([dotted_line_pos,dotted_line_pos],[1.0,0], 'k--',linewidth=3.0)


plt.legend(loc='lower left')
plt.xlabel('Masking genes along PC'+ str(PC+1))
plt.ylabel('Prediction accuracy(%)')
plt.title('Min2Max Primary Capsule'+ ' '+ str(Primary_capsule) +'-Type'+ str(Cell_type) )
#plt.show()

#scatter plot
plt.subplot(2,2,4)
plt.scatter(weightpca[:, 0], weightpca[:, 1],color='r', s=5,alpha=0.5,label = 'gene')
if dotted_line < 0:
    select_genes = []
np.save("Min2Max_genes.npy",np.asarray(select_genes))
plt.scatter(weightpca[select_genes, 0], weightpca[select_genes, 1], color = 'b',s=6,label = 'select_gene')
plt.legend(loc='lower left')
plt.ylabel('PC2', fontsize=10)
plt.xlabel('PC1', fontsize=10)
plt.title('Min2Max Primary Capsule'+ ' '+ str(Primary_capsule) +'-Type'+ str(Cell_type) )

plt.savefig("FE_Model_analysis_2_resutls.png")
plt.show()
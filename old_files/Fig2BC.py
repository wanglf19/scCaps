#! -*- coding: utf-8 -*-
# the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112
from Capsule_Keras import *
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras import utils
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

cell_type = ['B cells','CD14+ Monocytes','CD4+ T cells','CD8+ T cells','Dendritic Cells',
 'FCGR3A+ Monocytes','Megakaryocytes','Natural killer cells']
randoms = 30

data = np.load('data/PBMC_data.npy')
labels = np.load('data/PBMC_celltype.npy')

num_classes = 8

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

encoder1 = Model(x_in, x1)
encoder2 = Model(x_in, x2)
encoder3 = Model(x_in, x3)
encoder4 = Model(x_in, x4)
encoder5 = Model(x_in, x5)
encoder6 = Model(x_in, x6)
encoder7 = Model(x_in, x7)
encoder8 = Model(x_in, x8)
encoder9 = Model(x_in, x9)
encoder10 = Model(x_in, x10)
encoder11 = Model(x_in, x11)
encoder12 = Model(x_in, x12)
encoder13 = Model(x_in, x13)
encoder14 = Model(x_in, x14)
encoder15 = Model(x_in, x15)
encoder16 = Model(x_in, x16)

x = Concatenate()([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16])
x = Reshape((16, z_dim))(x)
capsule = Capsule(num_classes, 16, 3, False)(x)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule)

model = Model(inputs=x_in, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.load_weights('data/Demo_PBMC.weights')

#Fig 2 BC
##############################################################################################
encoder1weight = encoder1.get_weights()
encoder2weight = encoder2.get_weights()
encoder4weight = encoder4.get_weights()
encoder5weight = encoder5.get_weights()
encoder6weight = encoder6.get_weights()
encoder8weight = encoder8.get_weights()
encoder9weight = encoder9.get_weights()
encoder10weight = encoder10.get_weights()
encoder12weight = encoder12.get_weights()
encoder14weight = encoder14.get_weights()
encoder15weight = encoder15.get_weights()
encoder16weight = encoder16.get_weights()

encoder1weight0 = encoder1weight[0]
encoder2weight0 = encoder2weight[0]
encoder4weight0 = encoder4weight[0]
encoder5weight0 = encoder5weight[0]
encoder6weight0 = encoder6weight[0]
encoder8weight0 = encoder8weight[0]
encoder9weight0 = encoder9weight[0]
encoder10weight0 = encoder10weight[0]
encoder12weight0 = encoder12weight[0]
encoder14weight0 = encoder14weight[0]
encoder15weight0 = encoder15weight[0]
encoder16weight0 = encoder16weight[0]

totalweight =encoder4weight0
pca = PCA(n_components=16)
pca.fit(totalweight)
weightpca = pca.transform(totalweight)
color = [ 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:orange','tab:pink', 'tab:cyan', 'tab:olive']

#################################################################################################
#plot moving line for accuracy along PC

ratio_plot = np.zeros((8,61))
difference = (0.2903915 - (-0.205))/60
slice_line_pos = (0.2903915 - 0.0845)/difference

for j in range(61):
    #sub A select genes
    gene_count = 0
    for i in range(3346):
        if weightpca[i, 0] > (0.2903915 - difference*j):
            gene_count = gene_count + 1
            rownum = x_test.shape[0]
            x_test[:, i] = np.zeros(rownum)

    #sub B calculate the accuracy
    Y_pred = model.predict(x_test)
    Y_pred_order = np.argsort(Y_pred,axis=1)
    Y_pred_1 = Y_pred_order[:,7]

    current_type = 0
    total = [0,0,0,0,0,0,0,0]
    correct = [0,0,0,0,0,0,0,0]

    for i in range(x_test.shape[0]):
        index_int = int(Y_test[i])
        if Y_test[i] == Y_pred_1[i]:
            correct[index_int] = correct[index_int] + 1
        total[index_int] = total[index_int] + 1

    ratio_drop = [0,0,0,0,0,0,0,0]
    for i in range(len(total)):
        ratio_drop[i] = correct[i]/total[i]

    for i in range(len(total)):
        ratio_plot[i,j] = ratio_drop[i]

#module for scatter plot
################################################################################################################
removed = [[],[],[]]
for i in range(3346):
    if weightpca[i, 0] > (0.2903915 - difference *13):
        removed[0].append(i)
    if weightpca[i, 0] > (0.2903915 - difference *19) and weightpca[i, 0] <= (0.2903915 - difference *13):
        removed[1].append(i)
    if weightpca[i, 0] > (0.0845) and weightpca[i, 0] <= (0.2903915 - difference *19):
        removed[2].append(i)

plt.scatter(weightpca[:, 0], weightpca[:, 1],color='r', s=5,alpha=0.5)
plt.scatter(weightpca[removed[0], 0], weightpca[removed[0], 1], color = 'b',s=6)
plt.scatter(weightpca[removed[1], 0], weightpca[removed[1], 1], color = 'b',s=6)
plt.scatter(weightpca[removed[2], 0], weightpca[removed[2], 1], color = 'b',s=6)

cut_pos1 = 0.2903915 - difference *13
cut_pos2 = 0.2903915 - difference *19
cut_pos3 = 0.0845

plt.plot([cut_pos1,cut_pos1],[-0.13,0.13], 'k--',linewidth=3.0)
plt.plot([cut_pos2,cut_pos2],[-0.13,0.13], 'k-.',linewidth=3.0)
plt.plot([cut_pos3,cut_pos3],[-0.13,0.13], 'k:',linewidth=3.0)

plt.ylabel('PC2', fontsize=10)
plt.xlabel('PC1', fontsize=10)
plt.title('Primary capsule 4 (CD8+ T cells)')
plt.show()

#module for line plot
################################################################################################################
fig = plt.figure(figsize=(10,6))
for i in range(8):
    plt.plot(ratio_plot[i],  c=color[i], label=cell_type[i])

plt.plot([13,13],[1.0,0], 'k--',linewidth=3.0)
plt.plot([19,19],[1.0,0], 'k-.',linewidth=3.0)
plt.plot([np.round(slice_line_pos),np.round(slice_line_pos)],[1.0,0], 'k:',linewidth=3.0)

plt.legend(loc='lower left')
plt.xlabel('Masking genes along PC1')
plt.ylabel('Prediction accuracy(%)')
plt.title('Primary capsule 4 (CD8+ T cells)')
plt.show()



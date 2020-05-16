#! -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

randoms = 30

data = np.load('../data/PBMC_data.npy')
labels = np.load('../data/PBMC_celltype.npy')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test

y_train = np.asarray(y_train)
print(y_train.shape)

#Support Vector Machine
#########################################################################################################
svm_clf = LinearSVC()
svm_clf.fit(x_train, y_train)
svm_predict = svm_clf.predict(x_test)

count = 0
correct = 0
for i in range(len(svm_predict)):
    if svm_predict[i] == y_test[i]:
        correct = correct + 1
    count = count + 1
print(count)
print('svm ',correct/count)

#Random Forest
#########################################################################################################
RF_clf = RandomForestClassifier(n_estimators=50)
RF_clf.fit(x_train, y_train)
RF_predict = RF_clf.predict(x_test)

count = 0
correct = 0
for i in range(len(RF_predict)):
    if RF_predict[i] == y_test[i]:
        correct = correct + 1
    count = count + 1
print('RF ',correct/count)

#Linear Discriminant Analysis
#########################################################################################################
LDA_clf = LinearDiscriminantAnalysis()
LDA_clf.fit(x_train, y_train)
LDA_predict = LDA_clf.predict(x_test)

count = 0
correct = 0
for i in range(len(RF_predict)):
    if LDA_predict[i] == y_test[i]:
        correct = correct + 1
    count = count + 1
print('LDA ',correct/count)

#Nearest Neighbor
#########################################################################################################
knneigh = KNeighborsClassifier(n_neighbors=9)
knneigh.fit(x_train, y_train)
knn_predict = knneigh.predict(x_test)

count = 0
correct = 0
for i in range(len(RF_predict)):
    if knn_predict[i] == y_test[i]:
        correct = correct + 1
    count = count + 1
print('KNN ',correct/count)
## An interpretable deep-learning architecture of capsule networks for identifying cellular-type gene expression programs from single-cell RNA-seq data

This repository contains the official Keras implementation of:

**An interpretable deep-learning architecture of capsule networks for identifying cellular-type gene expression programs from single-cell RNA-seq data**



**Requirements**
- Python 3.6
- conda 4.4.10
- keras 2.2.4
- tensorflow 1.11.0


**Model training**

To unzip the PBMC_data.rar into dictionary 'data' then run:

```
old version
old_files/Model_Training.py  #specific for the PBMC dataset

new version
default:
python Model_Training.py

Augments:
'--inputdata', type=str, default='data/PBMC_data.npy', help='address for input data'
'--inputcelltype', type=str, default='data/PBMC_celltype.npy', help='address for celltype label'
'--num_classes', type=int, default=8, help='number of cell type'
'--randoms', type=int, default=30, help='random number to split dataset into training and testing set'
'--dim_capsule', type=int, default=16, help='dimension of the capsule'
'--num_capsule', type=int, default=16, help='number of the primary capsule'
'--batch_size', type=int, default=400, help='training parameters_batch_size'
'--epochs', type=int, default=10, help='training parameters_epochs'

python Model_Training.py --randoms 20 --dim_capsule 32
```

**Model analysis**

- *Demo*

The following documents (Jupyter Notebook) contain the codes for reproducing Figure in the main text.
```
demo_reproducing_figures.ipynb
```
In case you don't comfortable with Jupyter Notebook, you could copy the following file from the dictionary 'old_files' into current dictionary then run:
```
python Fig1CD_Fig2DE_Fig3.py
python Fig2BC.py
python Fig4.py
python Fig5D.py
```
**Comparison Model**

- SVM,RF,LDA,KNN
```
cd comparison_model
python Machine_Learning.py
```

- Neural Networks
```
cd comparison_model
python Neural_Networks.py
```


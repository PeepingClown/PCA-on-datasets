#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading data
train=pd.read_csv("E:\Datasets\Arabic\Train.csv")
submission=pd.read_csv("E:\Datasets\Arabic\Test.csv")

y_train=pd.read_csv("E:\Datasets\Arabic\TrainLab.csv")
X_train=train
X_submission=submission
"""y_train.head()
X_train.head()"""

import matplotlib.colors as cm
normalize=cm.Normalize(vmax=1, vmin=0)

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_result=pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)

print(X_train.shape)
print(pca_result.shape)

plt.scatter(pca_result[:4000,0], pca_result[:4000, 1],norm=normalize, c=y_train[:4000], edgecolor='none', alpha=0.5, cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()

pca=PCA(200)
pca_full=pca.fit(X_train)

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel("# of components")
plt.ylabel("cumulative explained variance")

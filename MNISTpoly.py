#importing libraries
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC


train_loc="E:\Datasets\MNIST\Train.csv"
test_loc="E:\Datasets\MNIST\Test.csv"
train_label="E:\Datasets\MNIST\TrainLab.csv"

train=pd.read_csv(train_loc)
X=train.as_matrix()
train_size= X.shape[0]
label=pd.read_csv(train_label)

y=label.values

test=pd.read_csv(test_loc).as_matrix()
test_size=test.shape[0]

#visualising random numbers
"""img_idx=np.random.randint(test_size, size=6)
fig=plt.figure()
for i in range(6):
    instance=test[img_idx[i]]
    assert(len(instance)==784)
    img=instance.reshape(28, 28)
    plt.subplot(230+i+1)
    plt.title(img_idx[i]+1)
    plt.axis("off")
    plt.imshow(img, cmap=cm.gray)"""
   
#train_test split
X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.7, random_state=50)

#using pca
pca=PCA(n_components=33, whiten=True)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
test= pca.transform(test)

svc=SVC(C=10, kernel="poly", degree="2", gamma="scale", class_weight="balanced", random_state=50)
svc.fit(X_train, y_train)
score=svc.score(X_test, y_test)
print("Score {0:.4f}".format(score))

labels= svc.predict(test)


   

    

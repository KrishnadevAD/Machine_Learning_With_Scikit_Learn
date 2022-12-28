
# Decisison tree visualization
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() 



# taking data set make blobs 

# make_blobs lee chai blobs banai dinxa 
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=300,centers=4,
random_state=0,
cluster_std=1)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='rainbow')
plt.show()
# from sklearn.tree import DecisionTreeClassifier
# tree=DecisionTreeClassifier()


# from sklearn.ensemble import  BaggingClassifier
# tree2= DecisionTreeClassifier()
# bag= BaggingClassifier(tree2,n_estimators=100,max_samples=0.8,random_state=1)
# bag.fit(X,y)
# visualize_classifier(bag,X,y)



# from sklearn.ensemble import RandomForestClassifier







def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
    clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
    np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
    plt.show()
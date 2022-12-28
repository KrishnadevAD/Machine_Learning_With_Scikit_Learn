# Gaussian Mixture Model
#Unsupervised learning Iris Dimentionality

# we first reduce the dimensionality of the Iris data to visualize there are 4 features
# The work of dimension redcution is to find suitable lower dimension that retains essential feature of data
# this helps in visualization of data

import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#getting the data set form the seaborn
iris_data = seaborn.load_dataset('iris')

#extraction of only the feature set
X_iris = iris_data.drop('species',axis=1)

#extraction of target set
y_iris = iris_data['species']


#Using PCA ( Principal Component Analysis )
#1 Choose the model
from sklearn.decomposition import PCA


#2 Instantiate with hyper parameter
#n_components states : how many dimension to change
model = PCA(n_components=2)


#3 Fit the data
model.fit(X_iris)


#4 Predict the data

# (method) transform(X: Any) -> Any
# Apply dimensionality reduction to X.
# X is projected on the first principal components previously extracted from a training set.
X_2D = model.transform(X_iris)


#Visualizing the data
iris_data['PCA1'] = X_2D[:,0]
iris_data['PCA2'] = X_2D[:,1]
#  fit_reg : (optional) This parameter accepting bool value . 
# If True, estimate and plot a regression model relating the x and y variables.
sns.lmplot(x="PCA1", y="PCA2", hue='species', data=iris_data, fit_reg=False)
plt.show()


# We see that in the two-dimensional representation, the species are fairly well separa‐ted, 
# even though the PCA algorithm had no knowledge of the species labels! 
# This indicates to us that a relatively straightforward classification will probably be effective
# on the dataset, as we saw before.

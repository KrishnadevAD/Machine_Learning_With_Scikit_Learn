# Principal Component Analysis (PCA)
# this is the unsupervised learning 

#imposting the useful libraries 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set()
iris = sns.load_dataset('iris')
x_iris=iris.drop(['species'],axis=1)
y_iris=iris['species']

# importing PCA ( principal components  analysis )
from sklearn.decomposition import PCA

# model insiansiate

# model fit garne ho 
#n_components=2 , kati ota dimendison  ma  divide garne 
model=PCA(n_components=2)
model.fit(x_iris)

# transform lee chai  4 dimensional  lai 2D ma pathauxa 
# 4 ota features lai chai calculation garer chai 2 ota maa chutyauxa hai 
# 4 ota bata ( 2 ota maa ), simply choose chai garne haina 

x_2D=model.transform(x_iris)

print(x_2D)

# visualization of 2D data 
iris['PCA1']=x_2D[:,0]
iris['PCA2']=x_2D[:,1]
sns.lmplot(x="PCA1",y="PCA2",hue='species',data=iris,fit_reg=False)
plt.show()




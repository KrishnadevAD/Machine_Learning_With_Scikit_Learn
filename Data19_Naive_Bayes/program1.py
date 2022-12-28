 # Gussian 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
from scipy.stats import norm
import math
#generation of simple data

def generatePlot(x):
    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-10, 10, 0.001)
    print(x_axis)
    # Mean = 0, SD = 2.
    mean = np.mean(x_axis)
    std = np.std(x_axis)
    plt.plot(x_axis, norm.pdf(x_axis, mean,std))
    plt.show()

#gettign dataset from sklear
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples = 100,n_features=2,centers=2,random_state=2,cluster_std=1.5)
#plotting the data
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='RdBu')
plt.show()
generatePlot(X)



#using naive bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X,y)

#generation of random new data blobs
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

#Plotting to get the aray of decison boundry
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()

#getting the probability of predicted t

yprob = model.predict_proba(Xnew)
print(yprob[-8:].round(2))

#SVM
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets._samples_generator import make_blobs

#Generation of data
X,y = make_blobs(n_samples = 50,centers = 2,random_state =0, cluster_std = 0.60)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plt.show()

#Linear classfication works my generating a line between the data
#for demonstration consider the following lines

#getting a linearly spaced data
xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
#adding a marker in a specific point
plt.plot([0.6],[2.1],'x',color='red',markeredgewidth=2,markersize=10)

##plotting a linear lines
for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b,'-k')
plt.xlim(-1,3.5)
plt.show()

#There are three different seperators for the same data that classifies data
#equally, However the value of X is very dependent on the line we choose


#TO solve this bring in SVM
#Rather then simply drawing a line we can draw margin of some width upto the
#nearest point


xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')

#(m,b,d) = values in tuple
for m,b,d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',color='#AAAAAA',alpha=0.4)
    plt.xlim(-1,3.5)
plt.show()

#In support vecotr machine the line with maximum margin is the one with optiamal
#model


#fititing svm
from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

#creating a function to visualize svm
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a two-dimensional SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)

    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)

plt.show()

# A support vector machine classifier fit to the data, with margins (dashed
# lines) and support vectors (line in maring )shown

# This is the dividing line that maximizes the margin between the two sets of points.
# Notice that a few of the training points just touch the margin; they are indicated by
# These points are the pivotal elements of this fit, and
# are known as the support vectors, and give the algorithm its name.
# In Scikit-Learn, the
# identity of these points is stored in the support_vectors_ attribute of the classifier

print(model.support_vectors_)

# A key to this classifier’s success is that for the fit, only the position of the support vec‐
# tors matters; any points further from the margin that are on the correct side do not
# modify the fit! Technically, this is because these points do not contribute to the loss
# function used to fit the model, so their position and number do not matter so long as
# they do not cross the margin



#Plotting for more data with mode data set similarly as above
def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
    random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))
plt.show()

#Incase of SVM the number of dataset does nt affect the model, is affected only by margin points seelecet or support vectors


# SVM with kernels
#
# Where SVM becomes extremely powerful is when it is combined with kernels.
#
# here we have data that are
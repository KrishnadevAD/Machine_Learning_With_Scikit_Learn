# SVM with kernels
#
# Where SVM becomes extremely powerful is when it is combined with kernels.
#
# here we have data that are

import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets._samples_generator import make_circles
from sklearn.svm import SVC


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

X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)
plt.show()

#here we can see that the linear kernel cannot seperat e the data
#we can solve this by changing data into higherr dimension
#On simple technice is to change the data is using radial basis function
r = np.exp(-(X ** 2).sum(1))

#Plotting this data in 3D
from mpl_toolkits import mplot3d
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

plot_3D()
plt.show()

# We can see that with this additional dimension, the data becomes trivially linearly
# separable, by drawing a separating plane at, say, r=0.7

# if we had not centered our radial basis function in the right location,
# we would not have seen such clean, linearly separable results

# One strategy to this end is to compute a basis function centered at every point in the
# dataset, and let the SVM algorithm sift through the results. This type of basis function
# transformation is known as a kernel transformation, as it is based on a similarity rela‐
# tionship (or kernel) between each pair of points

# A potential problem with this strategy—projecting N points into N dimensions—is
# that it might become very computationally intensive as N grows large. However,
# because of a neat little procedure known as the kernel trick, a fit on kernel-
# transformed data can be done implicitly—that is, without ever building the full N-
# dimensional representation of the kernel projection! This kernel trick is built into the
# SVM, and is one of the reasons the method is so powerful

#changing kerner to radial bias function
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)

#Plotting the data with changed kernel
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
s=300, lw=1, facecolors='none')
plt.show()


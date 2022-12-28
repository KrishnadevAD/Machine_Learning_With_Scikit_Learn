# #Bias/ Variance provides an appropriate to estimate the performance and also a way to select the model
# #A model that underfits the data and is less flexible is high bias model
# #A model that overfits the data and is more flexibleis high variance model
# # A high variance model will perform good on training data but fail on testing data
# # A high bias model will perform similarly on testing and training data, not necessasirily better
#
# #TO use validation curve in scikit we make use of polynimaial regression with degree as tunable param
# # We will use simple linear regression with pipe lines
#
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
#
# #Simply create a function that make polynomial regression using pipeline and linear reg


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))

import numpy as np
def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    #X.ravel() change 2D array to 1D
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y
X, y = make_data(40)

import matplotlib.pyplot as plt
import seaborn; 
seaborn.set() # plot formatting

X_test = np.linspace(-0.1, 1.1, 500)[:, None]
plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
#plt.show()

#first ma degree 1 liyer garyou ,
# mathi function maaa polynomial ma chai linear regression lagauxa

for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
plt.show()

#We can plot validation curve easily using the function provided by SCKITI
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,'polynomialfeatures__degree',degree, cv=7)
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()

# This shows precisely the qualitative behavior we expect: the training score is every‐
# where higher than the validation score; the training score is monotonically improving
# with increased model complexity; and the validation score reaches a maximum
# before dropping off as the model becomes overfit.


##From the validation curve we know that the optimal degree is 3

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.show()


#Learning curves
# One important aspect of model complexity is that the optimal model will generally
# depend on the size of your training data.

X2,y2 = make_data(200)
plt.scatter(X2.ravel(),y2)
plt.show()

degree = np.arange(0, 21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2,'polynomialfeatures__degree',degree, cv=7)
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score',linestyle ='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score',linestyle ='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()

# The solid lines show the new results, while the fainter dashed lines show the results of
# the previous smaller dataset. It is clear from the validation curve that the larger data‐
# set can support a much more complicated model
#Derived Features
#We have here a data that cannot be described by a straight line
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
plt.scatter(x,y)
plt.show()

#fit a simple linear regression

from sklearn.linear_model import LinearRegression
X = x[:,np.newaxis]
model = LinearRegression().fit(X,y)
yfit = model.predict(X)
plt.scatter(x,y)
plt.plot(x,yfit)
plt.show()

#it is clear that we need a more sophisticated model to fit this data and explain relationship between x and y
#we do this by adding extra column and transforming this data

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3,include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

#this derived column has one feature representing x another x2 and another x3
#now using linear regression on this

model = LinearRegression().fit(X2,y)
yfit = model.predict(X2)
plt.scatter(x,y)
plt.plot(x,yfit)
plt.show()

#we improved the model by intorducing a new column
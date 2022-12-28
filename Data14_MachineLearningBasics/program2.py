import matplotlib.pyplot as plt
import numpy as np


#-----------------Creation of Data
#creating a random range
# class numpy.random.RandomState(seed=None)¶
# Container for the Mersenne Twister pseudo-random number generator.
#rng--> RandomState(MT19937) at 0x7F5006D1FB40
rng = np.random.RandomState(42)

#creating x variable, i.e independent variable
#creating a independent varibale rng.rand generates 50 random variable between 0 and 1
x = 10 * rng.rand(50)
print("THe numpy array is :",np.array(x))
print("The shape of the numpy array is : ",np.array(x).shape)

#creating dpendent variable y, using x to create since they are dependent
y = 2 * x -1 + rng.rand(50)
print(y)
print(np.array(y).shape)

#plotting x and y in scatter plot to see relation
plt.scatter(x,y)
plt.show()

#-----------------End Creation of Data

#following the steps

#1 Choose a class of Model
#import the model that we want to use
from sklearn.linear_model import LinearRegression

#2 Chosing hyper parameter, hyper prameter is passed in class creation
#we have not yet applied the model, just storing the value

#fit_interceptbool, default=True
# Whether to calculate the intercept for this model. 
# If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
model = LinearRegression(fit_intercept=True)

#3 Arranging the data into feature and target vector
# we need to change the xvector into martix of size [n_samples,n_features]
# this can be done by reshaping the array
#using np.newaxis to change single dimension axis into multidimension axis
X = x[:,np.newaxis]
print("the shape of the new axis is :",X.shape)

#4 Fitting the model
#fitting the model applies the statistical algorithm and returns the data in calculated result in model
#object that can be used
model.fit(X,y)

#Interpreting result

#The sign of coeff tells whether there is +ve or -ve relation between variables
# A positive coefficient indicates that as the value of the independent variable increases,
# the mean of the dependent variable also tends to increase.
# A negative coefficient suggests that as the independent variable decreases,
# the dependent variable tends to decrease.
#the value tells us the magnitude of dependency or how strong is the dependency
print("Coeff: ",model.coef_)
#here the result between x varibale and y variable is +ve

#gives predicted value of y when x is 0
print("Intercept: ",model.intercept_)



#predicting value for unknown data
xfit = np.linspace(-1,11)

#changing to [nsamples,nfeatures]
Xfit = xfit[:,np.newaxis]
yfit = model.predict(Xfit)

#visualizing the data
plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()
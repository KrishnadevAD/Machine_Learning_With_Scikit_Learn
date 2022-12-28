#basic practice on the scikit-learn
#importing all the useful libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# here we have already give the data x and y to fit in the model 
#customer  linear Regression 
x= np.array([2,4,6,8,10,12])
y=np.array([1,2,3,4,5,6])

#visualization of the data
plt.scatter(x,y)
plt.show()


x_to_predict=np.array([24,56,36])

#imporing the liners regression algorithms
from sklearn.linear_model import LinearRegression

#initializing the model 
model = LinearRegression(fit_intercept=True)

# fitting the model 
X=x[:,np.newaxis] # converting 1D to multi_Dimensionsal
model.fit(X,y)

# fittting and printing the model coefficient , whether it is +ve or the -ve coefficient 
print("the model coefficient",model.coef_)
print("The model intercept ",model.intercept_)##gives predicted value of y when x is 0

#changing the shample into the n-shample and n-features 
Xfit=x_to_predict[:,np.newaxis]
Yfit=model.predict(Xfit)


# scatter plot
plt.scatter(X,y)
plt.plot(Xfit,Yfit)
plt.show()






# Restaurent Bill generation 
#impoting the all necessary  libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# x data 
bill_amount_x = np.array([2000,2340,2450,198,234,2500,1700,5000,3400])

# geneation ot the  y value  with the help of the x data 
# the  Tips in the restaurant is of the 3 %  of the bill amount
tips_y = (bill_amount_x*0.03).round(3)
print("The Tips in the restaurant are : ",tips_y)

# visualization of the data with the scatter plot 
plt.scatter(bill_amount_x,tips_y)
plt.show()

# data to predict 
data_to_predict= np.array([230,5000,678,9000])

#impoting the linear regression model 
from sklearn.linear_model import LinearRegression

# model 
model = LinearRegression(fit_intercept=True)

# changed bill 
#changing the bill amount becaure the bill amount is in only one dimension
# converitng the 1D to 2D
changed_bill_amount_x= bill_amount_x[:,np.newaxis]
model.fit(changed_bill_amount_x,tips_y)

#model fit
print("the model coefficient is : ",model.coef_)
print("the model intercept is ",model.intercept_)

#fitting the y set of the model 
bill_amount_x_fit= data_to_predict[:,np.newaxis]
y_fit= model.predict(bill_amount_x_fit)
print(y_fit)


#scatter plot of the model 
plt.scatter(changed_bill_amount_x,tips_y)
plt.plot(bill_amount_x_fit,y_fit)
plt.show()


# convering the bill data into the  csv file 
bill_data = pd.DataFrame({'bill amount': bill_amount_x, 'tips': tips_y})
bill_data.to_csv('bill.csv', index=True)
print("THe bill data are : ",bill_data)
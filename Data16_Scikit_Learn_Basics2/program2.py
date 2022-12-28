import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#creating an np array for bill amount
billAmount = np.array([2000,2340,2450,198,234,2500,1700,5000,3400])
print("Bill Amount Size",billAmount.shape)
# creating and np array of bill tips
billTips = billAmount * 0.03
print("Bill Tips Size ",billTips.shape)

#checking the plotted graph
plt.scatter(billAmount,billTips)
plt.show()

#1 Importing the algorithm
from sklearn.linear_model import LinearRegression

#2 Instantiation of model
model = LinearRegression(fit_intercept=True)

#3 Transofrmation of X data to [nsamples,nfeatures]
X = billAmount[:,np.newaxis]
print("New Bill Amount Shape",X.shape)

# Fitting the model
model.fit(X,billTips)

#Extaction of result
print("Coeff",model.coef_)
print("Intercept",model.intercept_)

#Prediction
dataToPredict = np.array([230,5000,678,9008])
X_predict = dataToPredict[:,np.newaxis]

predictedData = model.predict(X_predict)
print("Predicted Data",predictedData)

#exporting the prediction result
exportedDataFrame = pd.DataFrame({"Bill Amount":dataToPredict,"Tips":predictedData})
print(exportedDataFrame)
exportedDataFrame.to_csv('BillPredictionResult.csv',index=False)
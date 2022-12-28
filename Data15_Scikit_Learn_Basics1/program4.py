
# GaussianNB for the prediciton of the species of the iris dataset 
# # importing all the necessary libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# importing the iris data set 
sns.set()
iris_data_set = sns.load_dataset('iris')
print(iris_data_set)

print(iris_data_set.head())

print(iris_data_set.tail())

print("THe shape of the iris data set is : ",iris_data_set.shape)

print(iris_data_set.info())

print(iris_data_set.describe())


print("######################################")
# feature set
# here the species column of the iris data set is drop and the remaining is printed in the X_iris
X_iris = iris_data_set.drop('species', axis = 1)      # feature set
print(X_iris)


print("######################################")


# here the species of the of the iris data set is printed
y_iris= iris_data_set['species']
print(y_iris)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
#random_state=8 , differs the output and the accuracy of the model 
# higher value of the random_state the lower the accuracy and vice-versa
# here the random_state=1 the accuracy is almost nearer to the  100 ,
#  but the random state is changed in the next program4.py, have a look  hai  krishnadev !!!!!
# here the random state is   8 , the accuracy is less then that of the random state = 1
Xtrain,Xtest,ytrain,ytest = train_test_split(X_iris,y_iris, random_state=8)


# model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# fitting the model
model.fit(Xtrain, ytrain)


#ValueError: x and y must be the same size
# print(" the scatter plot ")
# plt.scatter(Xtrain,ytrain)
# plt.plot(Xtrain,ytrain)
# plt.show()

# prediction 
y_predicted = model.predict(Xtest)


# plotting the data with the  perdicted and test data 
plt.scatter(y_predicted,ytest)
plt.plot(y_predicted,ytest)
plt.show()

# just df maa rakheko data haru lai 
# keeping the data into the datframe , with predicted and actual data 
my_data = pd.DataFrame({'Predicted': y_predicted, 'Actual': ytest})
print(my_data)
# converting into the CSV file
my_data.to_csv("PredictionOfSpecies_iris.csv",index=False)

# #exporting the prediction result
# exportedDataFrame = pd.DataFrame({"Bill Amount":dataToPredict,"Tips":predictedData})
# print(exportedDataFrame)
# exportedDataFrame.to_csv('BillPredictionResult.csv',index=False)

#  Testing accuracy nikaleko 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ytest, y_predicted)
print('The accuracy is', accuracy*100)
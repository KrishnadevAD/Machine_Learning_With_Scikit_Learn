# k nearest neighbor(KNN) and Gaussian Naive bayes(GNB)

# importing all the necessary libraries 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set()

iris = sns.load_dataset('iris')

X_iris = iris.drop(['species'], axis =1 )# dropping the species of the iris dataset
Y_iris = iris['species']# species of the iris data set 



# KN neighbouring Algorithm
from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier(n_neighbors=2)
model1.fit(X_iris, Y_iris)
y_pred = model1.predict(X_iris)


# accuracy score 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_iris,y_pred)
print('The accuracy using accuracy_score is', accuracy*100)


#Model Holdout
# splitting the data into the tain test  using the train test split 
from sklearn.model_selection import train_test_split
# train_size=0.2, means among the 100% data the 20% data is used for the training of the data rest 80% is for test 
Xtrain,Xtest,ytrain,ytest = train_test_split(X_iris,Y_iris, random_state=1, train_size=0.2)   


# fitting the model 
model1.fit(Xtrain,ytrain)
y_pred2 = model1.predict(Xtest)


# accuracy score 
accuracy_2 = accuracy_score(ytest, y_pred2)
print('The accuracy using cross validation is', accuracy_2*100)


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB # imprting the GaussinaNB from sklearn.naive_baiyes
model_2 = GaussianNB()# Gaussian naive biayes chai , model2  lagayou 
model_2.fit(X_iris,Y_iris)# fitting the model2 


# cross validation of the models 
# cv = 2 means two times cross validate

#for model1
from sklearn.model_selection import cross_val_score
cross_val_KNN = cross_val_score(model1, X_iris, Y_iris, cv=2)   
print('The accuracy of KNN is', np.array(cross_val_KNN).mean())

#for model2
cross_val_GNB = cross_val_score(model_2,X_iris,Y_iris,cv=2)
print('The accuracy of GNB is', np.array(cross_val_GNB).mean())


# cross validation of the models 
# cv = 8 means 8 times cross validate

#for model1
from sklearn.model_selection import cross_val_score
cross_val_KNN = cross_val_score(model1, X_iris, Y_iris, cv=8)   
print('The accuracy of KNN is', np.array(cross_val_KNN).mean())

#for model2
cross_val_GNB = cross_val_score(model_2,X_iris,Y_iris,cv=8)
print('The accuracy of GNB is', np.array(cross_val_GNB).mean())
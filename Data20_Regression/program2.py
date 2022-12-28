#logistic Regression
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import math
import sklearn


titanic_data = pandas.read_csv(r'C:\Users\DELL\Desktop\data science DWIT\day20\titanic.csv')
print("NUmber of passengers",len(titanic_data.index))
print(titanic_data.keys())


#analyzing data to check how variable is affected  by another, relation between variable
sns.countplot(x="Survived",  data= titanic_data)
# plt.show()

#Checking survival of men and women
sns.countplot(x="Survived", hue='Sex', data= titanic_data)
# plt.show()

#Checking class of passengers travel
sns.countplot(x="Survived", hue='Pclass', data= titanic_data)
# plt.show()

#Checking null data and removing
print(titanic_data.isnull().sum())

titanic_data.drop('Cabin',axis=1,inplace=True)
print(titanic_data.isnull().sum())

#Cearnign other null data
titanic_data.dropna(inplace=True)
print(titanic_data.isnull().sum())

#change to categorical variable
sex = pd.get_dummies(titanic_data['Sex'],drop_first=True)
#getting only and dropping on column
print(sex)

embarked = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
print(embarked)

pclass = pd.get_dummies(titanic_data['Pclass'],drop_first=True)
print(pclass)

#aadding in data set
titanic_data = pd.concat([titanic_data,sex,embarked,pclass],axis=1)
print(titanic_data.head())

#removing categorical column
titanic_data.drop(columns=['Pclass','Sex','Embarked','PassengerId','Name','Ticket'],axis =1,inplace=True)
print(titanic_data.head())


#Trainign and testing
from sklearn.model_selection import train_test_split
X = titanic_data.drop('Survived',axis=1)
y = titanic_data['Survived']

train_x,test_x,y_train,y_test = train_test_split(X,y,random_state=1,train_size=0.5)

#Applycing model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x,y_train)
predY = model.predict(test_x)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predY,y_test)
print(accuracy)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,predY)
print(cf)

sns.heatmap(cf)
plt.show()
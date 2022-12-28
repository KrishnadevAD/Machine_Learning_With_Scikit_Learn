#feature Engineering
# Categorical feature engineering 
#The transformations of feature engineering may involve changing
#  the data representation or applying statistical methods to create new attributes (a.k.a. features).
#  One of the most common feature engineering methods for 
# categorical attributes is transforming each categorical attribute into a numeric representation


data=[
     
     {'price':85000,'rooms':4,'neighbourhood':'Queen Anne'},
     {'price':70000,'rooms':3,'neighbourhood':'Fremont'},
     {'price':65000,'rooms':2,'neighbourhood':'Welllingoford'},
     {'price':60000,'rooms':2,'neighbourhood':'Fremont'},
]
print("the data without any transform",data)


import pandas as pd

#DictVectorizer :Transforms lists of feature-value mappings to vectors.
from sklearn.feature_extraction import DictVectorizer 
vec=DictVectorizer(sparse=False,dtype=int)
newData=vec.fit_transform(data)
print("The data after the transformation are :",newData)


print("The vectorized features of the data are : ",vec.get_feature_names_out())


dataFrame=pd.DataFrame(newData,columns=vec.get_feature_names_out())
print("The dataFrame of the Vectorized data :",dataFrame)


vec2=DictVectorizer(sparse=True,dtype=int)# sparse=True--> sparse matrix maa chai value diyou yesle 
newData2=vec2.fit_transform(data)
print(newData2)


# text feature 
# count vectorizer lee euta word  ko feature cocunt gardinxa 
# use scikit learn ko count vectorizer 


#NOTE:
# AttributeError: 'DictVectorizer' object has no attribute 'get_feature_names'  will occurs in the program when we use
# the get_feature_names() in place of get_feature_names_out().
# but when we use the get_feature_names() in the google Colab than it will give no  any errors/attributeErrors

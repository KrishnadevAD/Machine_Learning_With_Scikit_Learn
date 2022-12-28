 # Categorical feature engineering 
import pandas as pd
data=[
     
     {'price':85000,'rooms':4,'neighbourhood':'Queen Anne'},
     {'price':70000,'rooms':3,'neighbourhood':'Fremont'},
     {'price':65000,'rooms':2,'neighbourhood':'Welllingoford'},
     {'price':68000,'rooms':8,'neighbourhood':'Fremont'},
      {'price':87000,'rooms':4,'neighbourhood':'Anne'},
     {'price':50000,'rooms':6,'neighbourhood':'Fremontee'},
     {'price':67000,'rooms':5,'neighbourhood':'Welllingofordking'},
     {'price':69000,'rooms':2,'neighbourhood':'Fremont king'},
 ]

from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False,dtype=int)# sparse=False , sparse matrix maa lagdaina hai 
newData=vec.fit_transform(data)
print(newData)

print(vec.get_feature_names_out())

dataFrame=pd.DataFrame(newData,columns=vec.get_feature_names_out())
print(dataFrame)

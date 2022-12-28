import numpy as np
data = [
    {'price':85000,'rooms':4,'neighborhood':'Jawalakhel'},
    {'price':70000,'rooms':3,'neighborhood':'Kopundole'},
    {'price':65000,'rooms':3,'neighborhood':'Kapan'},
    {'price':60000,'rooms':2,'neighborhood':'Kopundole'},
]

#we might encode the data straight forwardly this way

newData = {'Jawalakhel':1,'Kopundole':2,'Kapan':1}
# Sckiti models make the fundamental assumption that numerical features reflect algebraic
# quantities. Thus such a mapping would imply, for example, that Jawalakhel < Kopundole
# Tries to build a algebric equations

#one way to solve this is by using one-hot encoding which creates extra
# column indication presense of data or not using 1 or 0
#use scikit dict vectorizer for this process

from sklearn.feature_extraction import DictVectorizer
# A sparse matrix is a matrix in which many or most of the elements have a value of zero.
#sparse = true creates a sparse numpy matrix
vec = DictVectorizer(sparse=False,dtype=int)
newData = vec.fit_transform(data)

print(data)
#vectorized Data
print("Vectorized Data\n",newData)
print(vec.get_feature_names_out())

# There is one clear disadvantage of this approach: if your category has many possible
# values, this can greatly increase the size of your dataset. However, because the enco‐
# ded data contains mostly zeros, a sparse output can be a very efficient solution

vec2 = DictVectorizer(sparse=True,dtype=int)
#Sparse = true saves data by removing 0 data but keeping numerical feature intact
newData2 = vec2.fit_transform(data)
print(newData2)
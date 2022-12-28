#Text Features
# most automatic mining of social media data relies
# on some form of encoding the text as numbers.


import pandas as pd
sample = ['problem  of evil','evil queen','horizon problem']

#for vectorization of this data we construct a column representing word problem,evil and horizon
#A simple example can be to use Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x = vec.fit_transform(sample)
print(x)
# The result is a sparse matrix recording the number of times each word appears; it is
# easier to inspect if we convert this to a DataFrame with labeled columns:

#changing data to pandas data column, converting x = multidimension and passing columns as vectors feature names
data = pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())
print(data)

#count vectorizers gives simple count which many give to much words that appear frequently to solve this, we use tfid vectorizer

#Tfid vectorizer works by giving weight to the words by counting how many times they appear in a word

from sklearn.feature_extraction.text import TfidfVectorizer
vec2 = TfidfVectorizer()
# #first fitting
# vec2.fit(sample)
# #then transofrming
# x2 = vec2.transform(sample)
x2 = vec2.fit_transform(sample)
data2 = pd.DataFrame(x2.toarray(),columns=vec2.get_feature_names_out())
print(data2)

#gives weight which is more useful.

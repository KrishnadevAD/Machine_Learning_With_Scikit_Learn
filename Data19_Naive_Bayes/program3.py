#Nominal Naive Bais
#Multinominal naive bayes makes use of multinominal dist.
#simple rule of thumb it categorizes data as discrete that is with a count of occurance

#multinominal is used to classify text
# One place where multinomial naive Bayes is often used is in text classification, where
# the features are related to word counts or frequencies within the documents to be
# classified

import seaborn as sns
#importing data
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
#printing columns
print(data.target_names)

#selecting simple categories and getting training and testing data
categories = ['talk.religion.misc','soc.religion.christian','sci.space','comp.graphics']
train = fetch_20newsgroups(subset='train',categories=categories)
test = fetch_20newsgroups(subset='test',categories=categories)

#printing a sample from data
print(train.data[5])

#to use this data in machine learning we need to first vectorize it use tfid with pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

#first vectorizes and then fits data
model = make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data,train.target)
labels =model.predict(test.data)

# for la in labels:
#     print(categories[la])
#now we have predicted the labels we check this by using confusion matrix

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
mat = confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

##Now predicting data

def predict_category(s,train=train,model = model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

print(predict_category('sending a payload to ISS'))
#Facial Recognizations
import numpy as np
import matplotlib.pyplot as plt
# getting the data set
from sklearn.datasets import fetch_lfw_people
import seaborn as sns
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# Plotting the data
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()

# Image contains 3000 pixels and will take heavy load on system
# We reduce the pixels by using PCA and transfroming to 150 Components
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,random_state=42)

# Using grid search and cross validation to explore combination of paramenter
# Here we will adjust C (which controls the margin hardness) and gamma (which
# controls the size of the radial basis function kernel), and determine the best model:
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)

# Now with this cross-validated model, we can predict the labels for the test data

model = grid.best_estimator_
yfit = model.predict(Xtest)

# Looking at feew of the test images with predicted value
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()

# generation of classification report
from sklearn.metrics import classification_report

print(classification_report(ytest, yfit,target_names=faces.target_names))


#using confusion metrics to display data

from sklearn.metrics import  confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
xticklabels=faces.target_names,
yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
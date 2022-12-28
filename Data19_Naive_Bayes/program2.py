#pipeline
#handling missing data
from numpy import nan
import numpy as np

X = np.array([[ nan, 0, 3 ],
[ 3, 7, 9 ],
[ 3, 5, 2 ],
[ 4, nan, 6 ],
[ 8, 8, 1 ]])
y = np.array([14, 16, -1, 8, -5])

#to apply this to ML we need to impute or change Nan we can do this by using imputer

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
print(X2)

from sklearn.linear_model import LinearRegression
#Then
model = LinearRegression().fit(X2,y)
y2 = model.predict(X2)
print(y2)

#pipline
#suppose we need to join impute then polynomial and ten linear use piepline

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
model = make_pipeline(SimpleImputer(strategy='mean'),
PolynomialFeatures(degree=2),
LinearRegression())

model.fit(X, y) # X with missing values, from above
print(y)
print(model.predict(X))
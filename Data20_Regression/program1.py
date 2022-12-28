#Linear Regression Actual Code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# plt.rcParams['figure.figsize'] = (20.0,10.0)

#reading data
data = pd.read_csv(r'C:\Users\DELL\Desktop\data science DWIT\day20\headbrain.csv')
print(data.shape)
print(data.head())


#Getting X and Y
X = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values


#getting mean X and mean y
mean_x = np.mean(X)
mean_y = np.mean(y)

#total values
n = len(X)

#Calulation of numerator and denomerator of formula
num = 0
deno = 0

for i in range(n):
    num = num+ (X[i] - mean_x) * (y[i]-mean_y)
    deno = deno + (X[i]- mean_x) **2

b1 = num/deno
b0 = mean_y - (b1 * mean_x)

#Pint Coeff
print(b1,b0)

#Plotting the values
max_x = np.max(X) + 100
min_x = np.min(X) - 100

#Calculate lines values of x and y
x = np.linspace(min_x,max_x,1000)
y2 = b0+b1 * x

plt.plot(x,y2,color='#58b970', label = 'Regression Line')
plt.scatter(X,y,c='#ef5432',label = 'Scatter Plot')

plt.xlabel('Head Size')
plt.ylabel('Brain Weigh')
plt.legend()
plt.plot()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from src.discovery import preProcessing
import seaborn as sns
import matplotlib.pyplot as plt

data = preProcessing()

X = data.drop(['price', 'price_per_carat'], axis=1).copy()
y = data['price_per_carat'].copy()

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

xtrain.drop('carat', axis=1, inplace=True)
xtestCarat = xtest['carat'].copy()
xtest.drop('carat',axis=1, inplace=True)

model = RandomForestRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

sns.distplot(ytest * xtestCarat - predictions * xtestCarat)
print(np.sqrt(mean_squared_error(ytest * xtestCarat, predictions * xtestCarat)))
print(ytest.mean())
print(ytest.median())
plt.show()
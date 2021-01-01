import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from data_discovery_and_processing import pre_processing

diamonds = pre_processing()


# Modeling
# Create test set
X = diamonds.drop(['price', 'price_per_carat'], axis=1)
y = diamonds['price_per_carat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train.drop('carat', axis=1, inplace=True)
xtestCarat = X_test['carat'].copy()
X_test.drop('carat', axis=1, inplace=True)
print(X_train.columns)
# Polynomial regression
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
y_test *= xtestCarat

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test) * xtestCarat
print('Linear Regression RMSE: ', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test, predictions, cmap='coolwarm')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('Linear Regression predictions plot')
plt.show()

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
predictions = ridge.predict(X_test) * xtestCarat
print('Ridge Regression RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test, predictions, cmap='RdGy')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('Ridge Regression predictions plot')
plt.show()

# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)
predictions = lasso.predict(X_test) * xtestCarat
print('Lasso Regression RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test, predictions, cmap='PiYG')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('Lasso Regression predictions plot')
plt.show()

# ElasticNet Regression
elas_net = ElasticNet()
elas_net.fit(X_train, y_train)
predictions = elas_net.predict(X_test) * xtestCarat
print('ElasticNet Regression RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test, predictions, cmap='ocean')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('ElasticNet Regression predictions plot')
plt.show()

# Decision Tree Regressor
dec_tree = DecisionTreeRegressor()
dec_tree.fit(X_train, y_train)
predictions = dec_tree.predict(X_test) * xtestCarat
print('Decision Tree Regressor RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test, predictions, cmap='RdYlGn')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('Decision Tree Regressor predictions plot')
plt.show()

# Random Forest Regression
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
predictions = forest.predict(X_test) * xtestCarat
print('Random Forest Regression RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test * xtestCarat, predictions * xtestCarat, cmap='hot')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('Random Forest Regression predictions plot')
plt.show()
print('Random Forest Regression RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
# Visualize the relation between actual values and the predictions
plt.scatter(y_test * xtestCarat, predictions * xtestCarat, cmap='hot')
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.title('Random Forest Regression predictions plot')
plt.show()

sns.distplot(y_test - predictions)
plt.show()



import pandas as pd
#from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


test_scores = pd.read_csv('test_scores.csv')  # import file we need
chosen_state = 1337  # state to keep data consistent

x = test_scores.drop("math score", axis=1)  # axis 1 is for columns
y = test_scores["math score"]   # y is the target we want to predict

### The sizes of the neural networks
small = (1)
medium = (3, 3)
large = (10, 10, 10)
huge = (50, 40, 30, 20, 10, 3)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=chosen_state)  # split data into the training and the testing
# test on 20% of data and train on 80% of data

my_regressor = MLPRegressor(solver="lbfgs", hidden_layer_sizes=small,
                              random_state=chosen_state, max_iter=1000)  # only 1 neuron to train the dataset first
my_regressor.fit(x_train, y_train)  #

prediction = my_regressor.predict(x_test)

acc = my_regressor.score(x_test, y_test)
mse = mean_squared_error(y_test, prediction)
print("Accuracy for small", acc)
print("Mean squared error for small", mse)
print()


my_regressor = MLPRegressor(solver="lbfgs", hidden_layer_sizes=medium,
                              random_state=chosen_state, max_iter=1000)  # only 1 neuron to train the dataset first
my_regressor.fit(x_train, y_train)  #

prediction = my_regressor.predict(x_test)

acc = my_regressor.score(x_test, y_test)
mse = mean_squared_error(y_test, prediction)
print("Accuracy for medium", acc)
print("Mean squared error for medium", mse)
print()



my_regressor = MLPRegressor(solver="lbfgs", hidden_layer_sizes=large,
                              random_state=chosen_state, max_iter=1000)  # only 1 neuron to train the dataset first
my_regressor.fit(x_train, y_train)  #

prediction = my_regressor.predict(x_test)

acc = my_regressor.score(x_test, y_test)
mse = mean_squared_error(y_test, prediction)
print("Accuracy for large", acc)
print("Mean squared error for large", mse)
print()


my_regressor = MLPRegressor(hidden_layer_sizes=huge,
                              random_state=chosen_state, max_iter=1000)  # only 1 neuron to train the dataset first
my_regressor.fit(x_train, y_train)  #

prediction = my_regressor.predict(x_test)

acc = my_regressor.score(x_test, y_test)
mse = mean_squared_error(y_test, prediction)
print("Accuracy for huge", acc)
print("Mean squared error for huge", mse)

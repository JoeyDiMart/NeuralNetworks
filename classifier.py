from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

#  Test for even or odd

X = np.array([[int(b) for b in f"{i:04b}"] for i in
              range(16)])  # this creates a numpy array of all the bit combinations for W,X,Y,Z
y = np.array([bin(i).count("1") % 2 for i in
              range(16)])  # y is whether the total # of 1's, an even # of 1's returns 0, odd returns 1
chosen_state = 1337  # state to keep data consistent

my_classifier = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(1),
                              random_state=chosen_state, max_iter=1000)  # only 1 neuron to train the dataset
my_classifier.fit(X, y)

result = my_classifier.predict([[1, 1, 0, 0]])
result2 = my_classifier.predict([[1, 1, 1, 0]])
result3 = my_classifier.predict([[1, 0, 1, 0]])
print("Results for 3 test predictions (should be 010: ", result, result2, result3)

prediction = my_classifier.predict(X)
acc = accuracy_score(y, prediction)
print("1 neuron 1 layer Accuracy score:", acc)

mse = mean_squared_error(y, prediction)
print("1 neuron 1 layer Mean squared error", mse)
print("\n")

classifier2 = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(3),
                            random_state=chosen_state, max_iter=1000)  # 3 neurons to train the dataset
classifier2.fit(X, y)

result = classifier2.predict([[1, 1, 0, 0]])
result2 = classifier2.predict([[1, 1, 1, 0]])
result3 = classifier2.predict([[1, 0, 1, 0]])
print("Results for 3 test predictions (should be 010: ", result, result2, result3)

prediction = classifier2.predict(X)
acc = accuracy_score(y, prediction)
print("3 neurons 1 layer Accuracy score:", acc)

mse = mean_squared_error(y, prediction)
print("3 neurons 1 layer Mean squared error", mse)
print("\n")

classifier3 = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(10, 3),
                            random_state=chosen_state, max_iter=1000)  # 3 neurons to train the dataset
classifier3.fit(X, y)

result = classifier3.predict([[1, 1, 0, 0]])
result2 = classifier3.predict([[1, 1, 1, 0]])
result3 = classifier3.predict([[1, 0, 1, 0]])
print("Results for 3 test predictions (should be 010: ", result, result2, result3)

prediction = classifier3.predict(X)
acc = accuracy_score(y, prediction)
print("10 neurons then 3 neurons, 2 layers Accuracy score:", acc)

mse = mean_squared_error(y, prediction)
print("10 neurons then 3 neurons, 2 layers Mean squared error", mse)
print("\n")

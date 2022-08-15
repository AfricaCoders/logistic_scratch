from packages.modules import *

# Load dataset
data = pd.read_csv("data.csv", delimiter=",")


# delete last column with NaN values and replace M and B to 0 and 1 respectively.
data = data.drop(["Unnamed: 32"], axis=1)
data["diagnosis"] = data["diagnosis"].replace("M", 0)
data["diagnosis"] = data["diagnosis"].replace("B", 1)


# Splitting datasets into training and testing set.
X_train, Y_train , X_test, Y_test = train_test_split(data)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

num_iterations = 2000
learning_rate = 0.005
print_cost = True

dims = X_train.shape[0]
w, b = initialize_parameters(dims)
params, grads, costs = optimize(w, b, X_train, Y_train, learning_rate, num_iterations, print_cost)

w = params["w"]
b = params["b"]

Y_prediction_test = predict(w, b, X_test)
Y_prediction_train = predict(w, b, X_train)

if print_cost:
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

results = {
    "costs": costs,
    "Y_prediction_test": Y_prediction_test,
    "Y_prediction_train": Y_prediction_train,
    "w": w,
    "b": b,
    "learning_rate": learning_rate,
    "num_iterations": num_iterations
}







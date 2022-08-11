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

dims = X_train.shape[0]
w, b = initialize_parameters(dims)

print(w.shape)





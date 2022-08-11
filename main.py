import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("data.csv", delimiter=",")


# delete last column with NaN values and replace M and B to 0 and 1 respectively.
data = data.drop(["Unnamed: 32"], axis=1)
data["diagnosis"] = data["diagnosis"].replace("M", 0)
data["diagnosis"] = data["diagnosis"].replace("B", 1)
# print(data.info())
# print(data.head())




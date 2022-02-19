# 04-1 Logistic Regression

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

fish = pd.read_csv("https://bit.ly/fish_csv_data")

# print(fish.head())

# print(pd.unique(fish["Species"])) # upper lower alphabet is distinguished

fish_input = fish[["Weight", "Length", "Diagonal", "Height",
                   "Width"]].to_numpy()  # change left columns of Pandas data frame to Numpy array

# print(fish_input[:5])

fish_target = fish["Species"].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# data preprocessing
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# KNeighborsClassification
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
# print(kn.score(train_scaled, train_target))
# print(kn.score(test_scaled, test_target))
print(kn.classes_)
print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# check of Sigmoid(logistic) function graph
# z = np.arange(-5, 5, 0.1)
# phi = 1 / (1 + np.exp(-z))
# plt.plot(z, phi)
# plt.grid(True)
# plt.xlabel("z")
# plt.ylabel("phi")
# plt.show()

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(a)
# 03-1 K-Neighbors regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
#        21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
#        23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
#        27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
#        39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
#        44.0])
# perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
#        115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
#        150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
#        218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
#        556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
#        850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
#        1000.0])

# # plt.scatter(perch_length, perch_weight)
# # plt.xlabel("length(cm)")
# # plt.ylabel("weight(g)")
# # plt.title("Perch's length & weight")
#
# train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)
#
# # data set of sklearn library must be 2D dimension array, so data set of train and test input data is need to transformed from 1D array to 2D array
# train_input = train_input.reshape(-1,1)
# test_input = test_input.reshape(-1,1)
# # print(train_input.shape, test_input.shape)
#
# knr = KNeighborsRegressor()
#
# # train model by using train and target data set
# knr.fit(train_input, train_target)
#
# # print(len(train_input))
# # print(len(test_input))
# print("Result from test set is {0}.".format(knr.score(test_input, test_target)))
# print("Result from train set is {0}.".format(knr.score(train_input, train_target)))
#
# ''' coefficient of determination
# R^2 = sum(target - estimation)^2 / sum(target - average)^2 '''
#
# test_prediction = knr.predict(test_input)
# mae = mean_absolute_error(test_target, test_prediction)
# # print(mae)
#
# # overfitting means score from train is higher than score from test
# # underfitting means score from test is higher than score from train or both result are very low, usually derived from not enough data set
#
# # adjust number of neighbors points
# knr.n_neighbors = 3
#
# # train model again
# knr.fit(train_input, train_target)
# print("Result from train set adjusted number of neighbors from 5 to 3 is {0}".format(knr.score(train_input, train_target)))
# print("Result from test set adjusted number of neighbors from 5 to 3 is {0}".format(knr.score(test_input, test_target)))
#
# # quiz 2
# # knr2 = KNeighborsRegressor()
# #
# # x = np.arange(5,46,1).reshape(-1,1)
# #
# # for n in [1,5,10]:
# #        knr2.n_neighbors = n
# #        knr2.fit(train_input, train_target)
# #        prediction = knr2.predict(x)
# #
# #        plt.scatter(train_input, train_target)
# #        plt.plot(x, prediction)
# #        plt.title("n_neighbors = {0}.".format(n))
# #        plt.xlabel("length")
# #        plt.ylabel("weight")
# #        plt.show()
#
# # 03-2 linear regression
# knr2 = KNeighborsRegressor(n_neighbors = 3)
#
# knr2.fit(train_input, train_target)
#
# distances, indexes = knr2.kneighbors([[50]])
#
# # plt.scatter(train_input, train_target)
# # plt.scatter(train_input[indexes], train_target[indexes], marker = "D")
#
# # 50cm perch data
# # plt.scatter(50, 1033, marker = "^")
# # plt.xlabel("length(cm)")
# # plt.ylabel("weight(g)")
#
# # print(np.mean(train_target[indexes]))
#
# lr = LinearRegression()
# lr.fit(train_input, train_target)
#
# print(lr.predict([[50]]))
#
# # intercept and coefficient
# print(lr.coef_, lr.intercept_)
#
# # scatter plot of given data set
# # plt.scatter(train_input, train_target)
#
# # 1D equation of variable from 15 to 50
# # plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
#
# # 50cm perch data
# # plt.scatter(50, 1241.8, marker = "^")
# # plt.xlabel('length(cm)')
# # plt.ylabel('weight(g)')
#
# print(lr.score(train_input, train_target))
# print(lr.score(test_input, test_target))
#
# # polynomial regression
#
# train_poly = np.column_stack((train_input**2, train_input))
# test_poly = np.column_stack((test_input**2, test_input))
#
# lr_2p = LinearRegression()
# lr_2p.fit(train_poly, train_target)
#
# # print(lr_2p.predict([[50**2, 50]]))
# # print(lr_2p.coef_, lr_2p.intercept_)
#
# point = np.arange(15,50)
# plt.scatter(train_input, train_target)
# plt.plot(point, lr_2p.coef_[0]*point**2 + lr_2p.coef_[1]*point + lr_2p.intercept_)
#
# # 50cm perch data
# plt.scatter(50, lr_2p.predict([[50**2, 50]]), marker = "^")
# plt.xlabel('length(cm)')
# plt.ylabel('weight(g)')
# plt.show()
#
# print(lr_2p.score(train_poly, train_target))
# print(lr_2p.score(test_poly, test_target))

# 03-3 Feature engineering and regularization
# get perch data of length from read_csv
df = pd.read_csv("https://bit.ly/perch_csv_data")
perch_full = df.to_numpy()

# get perch weight data
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])

# divide given data to train and test set
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

test_poly = poly.transform(test_input)

lr_poly = LinearRegression()
lr_poly.fit(train_poly, train_target)

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly5 = poly.transform(train_input)
test_poly5 = poly.transform(test_input)
# print(train_poly5.shape)
# print(poly.get_feature_names())

lr_poly.fit(train_poly5, train_target)
print(lr_poly.score(train_poly5, train_target))
print(lr_poly.score(test_poly5, test_target))

ss = StandardScaler()
ss.fit(train_poly5)
train_scaled = ss.transform(train_poly5)
test_scaled = ss.transform(test_poly5)

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("Result from ridge regularization(train) : {0}".format(ridge.score(train_scaled, train_target)))
print("Result from ridge regularization(test) : {0}".format(ridge.score(test_scaled, test_target)))

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel("length")
# plt.ylabel("R^2")

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print("Result from ridge regularization(train) with accurate alpha : {0}".format(ridge.score(train_scaled, train_target)))
print("Result from ridge regularization(test) with accurate alpha : {0}".format(ridge.score(test_scaled, test_target)))

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print("Result from lasso regularization(train) : {0}".format(lasso.score(train_scaled, train_target)))
print("Result from lasso regularization(test) : {0}".format(lasso.score(test_scaled, test_target)))

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel("length")
# plt.ylabel("R^2")

lasso = Lasso(alpha=1)
lasso.fit(train_scaled, train_target)
print("Result from lasso regularization(train) with accurate alpha : {0}".format(ridge.score(train_scaled, train_target)))
print("Result from lasso regularization(test) with accurate alpha : {0}".format(ridge.score(test_scaled, test_target)))

print(np.sum(lasso.coef_ == 0))
import pickle

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# ՄՈՒՏՔԱՅԻՆ ՏՎՅԱԼՆԵՐԻ ՖԱՅԼԸ, ՈՐՏԵՂ ՏՐՎԱԾ Է X և Y ԱՌԱՆՑՔՆԵՐԻ ԿՈՐԴԻՆԱՏՆԵՐԸ
input_file = 'data_singlevar_regr.txt'

# ՏՎՅԱԼՆԵՐԻ ՆԵՐՄՈՒԾՈՒՄ ԵՐԿՉԱՓ ԶԱՆԳՎԱԾԻ ՏԵՍՔՈՎ
data = np.loadtxt(input_file, delimiter=',')
"""
[[-0.86  4.38]
 [ 2.58  6.97]
 [ 4.17  7.01]
 [ 2.6   5.44]
 .............
"""

# ԱՌԱՆՁՆԱՑՆԵՆՔ X-Ի(ՏՈՂԵՐԻ 1-ԻՆ ՏԱՐՐԵՐԸ) և Y-Ի(ՏՈՂԵՐԻ 2-ՐԴ ՏԱՐՐԵՐԸ) ԿՈՐԴԻՆԱՏՆԵՐԸ
X, y = data[:, :-1], data[:, -1]
"""
X` 
[[-0.86]
 [ 2.58]
 [ 4.17]
 [ 2.6 ]
 [ 5.13]
 [ 3.23]
 .......]

Y` [4.38,  6.97,  7.01,  5.44, ... ]
"""

# ԲԱԺԱՆԵՆՔ ՏՎՅԱԼՆԵՐԸ ՈՒՍՈՒՑՄԱՆ և ԹԵՍՏԱՎՈՐՄԱՆ ՄԱՍԵՐԻ
num_training = int(0.8 * len(X))  # 80%
num_test = len(X) - num_training

# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

# ՍՏԵՂԾՈՒՄ ԵՆՔ ԳԾԱՅԻՆ ՌԵԳՐԵՍԻԱՅԻ ՄՈԴԵԼ
regressor = linear_model.LinearRegression()

# ՄՈԴԵԼԻ ՈՒՍՈՒՑՈՒՄԸ fit ՖՈՒՆԿՑԻԱՅԻ ՄԻՋՈՑՈՎ, ՕԳՏԱԳՈՐԾԵԼՈՎ training data-ՆԵՐԸ
regressor.fit(X_train, y_train)

# ԿԱՆԽԱՏԵՍՈՒՄ X-Ի ԹԵՍՏԱՅԻՆ ՏՎՅԼԱՆԵՐԻ ՀԻՄԱՆ ՎՐԱ
y_test_pred = regressor.predict(X_test)

# ԳՐԱՖԻԿԻ ԿԱՌՈՒՑՈՒՄ
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# ՀԱՇՎԵԼ ՉԱՓՈՒՄՆԵՐԻ ՃՇՏՈՒԹՅՈՒՆԸ
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# ՊԱՀՊԱՆԵՆՔ ՄՈԴԵԼԸ
with open('model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# ՆԵՐԱԾԵՆՔ ՄՈԴԵԼԸ
with open('model.pkl', 'rb') as f:
    regressor_model = pickle.load(f)

# ԿԱՏԱՐԵՆՔ ՆՈՐ ԿԱՆԽԱՏԵՍՈՒՄ ԹԵՍՏԱՅԻՆ ՏՎՅԱԼՆԵՐԻ ՀԻՄԱՆ ՎՐԱ ԱՐԴԵՆ ՆԵՐԱԾՎԱԾ ՄՈԴԵԼՈՎ
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))


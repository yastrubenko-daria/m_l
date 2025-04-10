# наивная байесовская классификация
# Набор моделей, которые предлагают быстрые и простые алгоритмы классификации

# гауссовский наивный байесовский классификатор
# наивное допущение состоит в том, что ! данные всех категорий взяты из простого нормального распределения

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

iris = sns.load_dataset('iris')
# print(iris.head())

# sns.pairplot(iris, hue='species')

data = iris[['sepal_length', 'petal_length', 'species']]
# print(data.head())
# print(data.shape)
# setosa versicolor - 2 мерное распределение

# data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]
# print(data_df.shape)
#
# X = data_df[['sepal_length', 'petal_length']]
# y = data_df['species']
# model = GaussianNB()
# model.fit(X, y)
#
# print(model.theta_[0])  # среднее мат ожидание
# print(model.var_[0])  # разброс
# print(model.theta_[1])
# print(model.var_[1])
#
# theta0 = model.theta_[0]
# var0 = model.var_[0]
# theta1 = model.theta_[1]
# var1 = model.var_[1]

# data_df_seposa = data_df[data_df['species'] == 'setosa']
# data_df_versicolor = data_df[data_df['species'] == 'versicolor']
#
# plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
# plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])
#
# x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
# x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)
#
# X1_p, X2_p = np.meshgrid(x1_p, x2_p)
# X_p = pd.DataFrame(
#     np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
# )
# print(X_p.head())
#
# z1 = 1 / (1 * np.pi * (var0[0] * var0[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta0[0]) ** 2 / var0[0] + (X2_p-theta0[1])**2/var0[1]))
#
# z2 = 1 / (1 * np.pi * (var1[0] * var1[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta1[0]) ** 2 / var1[0] + (X2_p-theta1[1])**2/var1[1]))
#
# plt.contour(X1_p, X2_p, z1)
# plt.contour(X1_p, X2_p, z2)
#
# y_p = model.predict(X_p)
# X_p['species'] = y_p
#
# X_p_setosa = X_p[X_p['species'] == 'setosa']
# X_p_versicolor = X_p[X_p['species'] == 'versicolor']
#
# print(X_p.head())
#
# plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
# plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)
#
# # sns.pairplot(data_df, hue='species')
#
#
# # setjsa virginica
#
# fig = plt.figure()
#
# ax = plt.axes(projection ='3d')
# ax.contour(X1_p, X2_p, z1, 40)
# ax.contour(X1_p, X2_p, z2, 40)


data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]
print(data_df.shape)

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']
model = GaussianNB()
model.fit(X, y)

print(model.theta_[0])  # среднее мат ожидание
print(model.var_[0])  # разброс
print(model.theta_[1])
print(model.var_[1])

theta0 = model.theta_[0]
var0 = model.var_[0]
theta1 = model.theta_[1]
var1 = model.var_[1]

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)
print(X_p.head())

z1 = 1 / (1 * np.pi * (var0[0] * var0[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta0[0]) ** 2 / var0[0] + (X2_p-theta0[1])**2/var0[1]))

z2 = 1 / (1 * np.pi * (var1[0] * var1[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta1[0]) ** 2 / var1[0] + (X2_p-theta1[1])**2/var1[1]))

plt.contour(X1_p, X2_p, z1)
plt.contour(X1_p, X2_p, z2)

y_p = model.predict(X_p)
X_p['species'] = y_p

X_p_virginica = X_p[X_p['species'] == 'virginica']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

print(X_p.head())

plt.scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha=0.2)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)

# sns.pairplot(data_df, hue='species')


# setjsa virginica

fig = plt.figure()

ax = plt.axes(projection ='3d')
ax.contour(X1_p, X2_p, z1, 40)
ax.contour(X1_p, X2_p, z2, 40)




plt.show()

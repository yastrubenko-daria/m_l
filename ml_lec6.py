# юорьюа с Переобучением
# переобучение присуще всем деревьям принятия решений
#

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
#
# iris = sns.load_dataset('iris')
#
# # sns.pairplot(iris, hue='species')
#
# species_int = []
# for r in iris.values:
#     match r[4]:
#         case 'setosa':
#             species_int.append(1)
#         case 'versicolor':
#             species_int.append(2)
#         case 'virginica':
#             species_int.append(3)
#
# species_int_df = pd.DataFrame(species_int)
#
# data = iris[['sepal_length', 'petal_length']]
# data['species'] = species_int
#
# data_versicolor = data[data['species'] == 2]
# data_virginica = data[data['species'] == 3]
#
# data_versicolor_A = data_versicolor.iloc[:25, :]
# data_versicolor_B = data_versicolor.iloc[25:, :]
#
# data_virginica_A = data_virginica.iloc[:25, :]
# data_virginica_B = data_virginica.iloc[25:, :]
#
# data_df_A = pd.concat([data_virginica_A, data_versicolor_A], ignore_index=True)
# data_df_B = pd.concat([data_virginica_B, data_versicolor_B], ignore_index=True)
#
# x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']), 100)
# x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']), 100)
#
# X1_p, X2_p = np.meshgrid(x1_p, x2_p)
# X_p = pd.DataFrame(
#     np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
# )
#
#
# fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')
# max_depth = [1,3,5,7]
#
# X = data_df_A[['sepal_length', 'petal_length']]
# y=data_df_A['species']
#
# j=0
# for md in max_depth:
#     model = DecisionTreeClassifier(max_depth=md)
#     model.fit(X, y)
#
#     ax[0,j].scatter(data_virginica_A['sepal_length'], data_virginica_A['petal_length'])
#     ax[0, j].scatter(data_versicolor_A['sepal_length'], data_versicolor_A['petal_length'])
#
#     y_p = model.predict(X_p)
#     ax[0,j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow',
#                      zorder=1)
#     j+=1
#
# X = data_df_B[['sepal_length', 'petal_length']]
# y=data_df_B['species']
#
# j=0
# for md in max_depth:
#     model = DecisionTreeClassifier(max_depth=md)
#     model.fit(X, y)
#
#     ax[1,j].scatter(data_virginica_B['sepal_length'], data_virginica_B['petal_length'])
#     ax[1, j].scatter(data_versicolor_B['sepal_length'], data_versicolor_B['petal_length'])
#
#     y_p = model.predict(X_p)
#     ax[1,j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow',
#                      zorder=1)
#     j+=1
#
#
# plt.show()


# Ансамблевые методы. В основе идея объединения нескольких переобученых (!) моделей для уменьшения эффекта переобучения
# Это называется баггинг (bagging)
# баггинг усредняет результаты -> оптимальной классификайии

# ансамбль случайных деревьев называется случайным лесом


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
iris = sns.load_dataset('iris')

# sns.pairplot(iris, hue='species')

# species_int = []
# for r in iris.values:
#     match r[4]:
#         case 'setosa':
#             species_int.append(1)
#         case 'versicolor':
#             species_int.append(2)
#         case 'virginica':
#             species_int.append(3)
#
# species_int_df = pd.DataFrame(species_int)
#
# data = iris[['sepal_length', 'petal_length']]
# data['species'] = species_int
#
# data_setosa = data[data['species'] == 1]
# data_versicolor = data[data['species'] == 2]
# data_virginica = data[data['species'] == 3]
#
# x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']), 100)
# x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']), 100)
#
# X1_p, X2_p = np.meshgrid(x1_p, x2_p)
# X_p = pd.DataFrame(
#     np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
# )
#
#
# fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
#
# ax[0].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
# ax[0].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])
# ax[0].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
#
# # max_depth = [1,3,5,7]
# #
# md=6
#
# X = data[['sepal_length', 'petal_length']]
# y=data['species']
#
# model1 = DecisionTreeClassifier(max_depth=md)
# model1.fit(X, y)
#
# y_p = model1.predict(X_p)
#
# ax[0].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow',
#                      zorder=1)
#
#
# #Bagging
# ax[1].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
# ax[1].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])
# ax[1].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
#
# model2 = DecisionTreeClassifier(max_depth=md)
# b=BaggingClassifier(model2, n_estimators=20, max_samples=0.8, random_state=1)
# b.fit(X, y)
#
# y_p2 = b.predict(X_p)
#
# ax[1].contourf(X1_p, X2_p, y_p2.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow',
#                      zorder=1)
#
#
# #RandomForestClassifier
#
# ax[2].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
# ax[2].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])
# ax[2].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
#
# model3 = RandomForestClassifier(n_estimators=20, max_samples=0.8, random_state=1)
#
# model3.fit(X, y)
#
# y_p3 = model3.predict(X_p)
#
# ax[2].contourf(X1_p, X2_p, y_p3.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow',
#                      zorder=1)



#Регрессия с помощью случайных весов



# from sklearn.ensemble import RandomForestRegressor
# data = iris[['sepal_length', 'petal_length', 'species']]
#
#
# data_setosa = data[data['species'] == 'setosa']
#
#
# x_p = pd.DataFrame(np.linspace(min(data_setosa['sepal_length']), max(data_setosa['sepal_length']), 100))
#
#
# X = pd.DataFrame(data_setosa['sepal_length'], columns=['sepal_length'])
# y=data_setosa['petal_length']
#
# model = RandomForestRegressor(n_estimators=20)
# model.fit(X, y)
#
# y_p = model.predict(x_p)
#
# plt.scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
# plt.plot(x_p,y_p)

#Достоинства
#-простота и быстрота. Возсожно распарелеливание процесса-> выигрыш во времени
# - Вероятностная классификация
# - Модель непараметрическая. Заранее не знаем что лежит в системе=> хорошо работает с задачами, где другие модели могут оказатся недоучинными
#
#Недостатки
# - сложно интерпретировать
#
#plt.show()


# Метод главных компонент
# PCA (principal component analysis) - алгоритм обучения без учителя
# PCA - часто используют для понижения размерности

#Задача машинного обучения БЕЗ учителя состоит в выяснения зависимости между признаками
# в PCA выполняется качественная оценка этой зависимости путем поиска главных осей координат и их использования для описания наборов данных

from sklearn.decomposition import PCA

iris = sns.load_dataset('iris')

#sns.pairplot(iris, hue='species')

data = iris[['petal_width', 'petal_length', 'species']]


data_v = data[data['species'] == 'versicolor']
data_v = data_v.drop(columns=['species'])
X =data_v['petal_width']
Y= data_v['petal_length']
plt.scatter(X,Y)

p= PCA(n_components=2)
p.fit(data_v)


plt.scatter(p.mean_[0], p.mean_[1])#- две оси
plt.plot(
    [p.mean_[0], p.mean_[0]+p.components_[0][0]*np.sqrt(p.explained_variance_[0])],
          [p.mean_[1], p.mean_[1]+p.components_[0][1]*np.sqrt(p.explained_variance_[0])]
)
plt.plot(
    [p.mean_[0], p.mean_[0]+p.components_[1][0]*np.sqrt(p.explained_variance_[1])],
          [p.mean_[1], p.mean_[1]+p.components_[1][1]*np.sqrt(p.explained_variance_[1])]
)



p1= PCA(n_components=1)
p1.fit(data_v)

X_p = p1.transform(data_v)
X_p_new = p1.inverse_transform(X_p)

plt.scatter(X_p_new[:,0], X_p_new[:,1], alpha=0.2)# спроицировали на главную ось

# + простота интерпретации, эффективность в работе с многомерными данными
# - Аномальные значения в данных оказывают сильное влияние






plt.show()













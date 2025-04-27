import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'petal_length', 'species']]

data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

data_df_seposa=data_df_seposa.iloc[round(len(data_df_seposa)/3):]
data_df_versicolor=data_df_versicolor.iloc[round(len(data_df_versicolor)/3):]

X=pd.concat([data_df_seposa, data_df_versicolor])[['sepal_length', 'petal_length']]
y = pd.concat([data_df_seposa, data_df_versicolor])['species']
print(X)

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)
print(model.support_vectors_)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=400, facecolor='none',
            edgecolors='black')  # точки-опорные вектора

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)
y_p = model.predict(X_p)
X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

print(X_p.head())

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)

## дз1: на том же наборе убрать часть данных, и убедиться что влияют только опорные вектора
plt.show()
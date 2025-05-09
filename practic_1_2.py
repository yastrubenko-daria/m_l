import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'sepal_width', 'petal_width', 'species']]

data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'sepal_width', 'petal_width']]
y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_df[['sepal_length', 'sepal_width', 'petal_width']])

model = PCA(n_components=3)
components = model.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    data=components,
    columns=['PC1', 'PC2', 'PC3']
)
pca_df['species'] = data_df['species'].values

fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                    color='species', )

fig.show()

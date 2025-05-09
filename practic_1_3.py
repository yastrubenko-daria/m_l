import plotly.express as px
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'sepal_width','petal_width', 'species']]

data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'sepal_width','petal_width']]
y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

model = KMeans(n_clusters=2)
model.fit(X)


x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 30)
x2_p = np.linspace(min(data_df['sepal_width']), max(data_df['sepal_width']), 30)
x3_p = np.linspace(min(data_df['petal_width']), max(data_df['petal_width']), 30)

X1_p, X2_p, X3_p = np.meshgrid(x1_p, x2_p, x3_p)
X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel(),X3_p.ravel()]).T, columns=['sepal_length', 'sepal_width','petal_width' ]
)
y_p = model.predict(X_p[['sepal_length', 'sepal_width', 'petal_width']])
X_p['species'] = y_p


X_p_setosa = X_p[X_p['species'] == 0]
X_p_versicolor = X_p[X_p['species'] == 1]


fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=data_df_seposa["sepal_length"],
    y=data_df_seposa["sepal_width"],
    z=data_df_seposa["petal_width"],
    mode="markers",
    marker=dict(
        size=5,
        color='blue',
        opacity=0.8
    ),
    name="Setosa"
)
)

fig.add_trace(go.Scatter3d(
    x=data_df_versicolor["sepal_length"],
    y=data_df_versicolor["sepal_width"],
    z=data_df_versicolor["petal_width"],
    mode="markers",
    marker=dict(
        size=5,
        color='red',
        opacity=0.8
    ),
    name="Versicolor"
)
)

fig.add_trace(go.Scatter3d(
    x=X_p_setosa["sepal_length"],
    y=X_p_setosa["sepal_width"],
    z=X_p_setosa["petal_width"],
    mode="markers",
    marker=dict(size=3, color='green', opacity=0.1),
    name="Setosa predict"
)
)
fig.add_trace(go.Scatter3d(
    x=X_p_versicolor["sepal_length"],
    y=X_p_versicolor["sepal_width"],
    z=X_p_versicolor["petal_width"],
    mode="markers",
    marker=dict(size=3, color='orange', opacity=0.1),
    name="Versicolor predict"
)
)

fig.update_layout(
    scene=dict(
        xaxis_title='sepal_length',
        yaxis_title='sepal_width',
        zaxis_title='petal_width',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
    ),
    legend_title='Legend',
    #margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()


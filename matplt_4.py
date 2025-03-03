import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#трехмерные точки и линии

fig=plt.figure()
ax=plt.axes(projection='3d')

z1=np.linspace(0,15,1000)
y1=np.cos(z1)
x1=np.sin(z1)

#ax.plot(x1,y1,z1, 'green')

z2=15*np.random.random(100)
y2=np.cos(z2)+0.1*np.random.random(100)
x2=np.sin(z2)+0.1*np.random.random(100)

ax.scatter3D(x2,y2,z2,c=z2, cmap='Greens')
def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))

x=np.linspace(-6,6,30)
y=np.linspace(-6,6,30)
X, Y =np.meshgrid(x,y)
Z=f(X,Y)

ax.contour3D(X, Y, Z, 40, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(45, 60)#отнсительно плоскости xy, z

ax.scatter3D(X,Y,Z,c=Z, cmap='Greens')

#каркасный
ax.plot_wireframe(X,Y,Z)

#поверхностый
ax.plot_surface(X,Y,Z,cmap='viridis', edgecolor='none')
ax.set_titel('Example')

#полярные координаты и срезы

r=np.linspace(0,6,20)
theta=np.linspace(-0.9*np.pi, 0.8*np.pi, 40)
R, Theta = np.meshgrid(r,theta)
X=r*np.sin(Theta)
Y=r*np.cos(Theta)

Z=f(X,Y)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis', edgecolor='none')

theta=2*np.pi+np.random.random(1000)
r=6*np.random.random(1000)

x=r*np.sin(theta)
y=r*np.cos(theta)
z=f(x,y)

ax.scatter(x,y,z,c=z, cmap='viridis')

ax.plot_trisurf(x,y,z,c=z,cmap='viridis')# треугольники дорисовывает

#Seaborn

# - DataFrame(Matplotlib c Pandas)
# - более высокоуровневый

data = np.random.multivariate_normal([0,0], [[5,2],[2,2]], size=2000)
data=pd.DataFrame(data, columns=['x', 'y'])

print(data.head())

fig=plt.figure()
plt.hist(data['x'], alpha=0.5)
plt.hist(data['y'], alpha=0.5)

fig = plt.figure()
sns.kdeplot(data=data, shade=True)#, x='x', y='y')

iris = sns.load_dataset('iris')
print(iris.head())

sns.pairplot(iris, hue='species', height=2.5) #по парам сравнивает числа


tips = sns.load_dataset('tips')
print(tips.head())

#гистограммы
grid= sns.FacetGrid(tips,row='day', col='sex', hue='time' )
grid.map(plt.hist, 'tip', bins=np.linspace(0,40,15))

#графики факторов
sns.catplot(data=tips, x='day', y='total_bill', kind='box')

sns.jointplot(data=tips, x='tip', y='total_bill', kind='hex') #столбики с усиками

planets = sns.load_dataset('planets')
print(planets.head())

sns.catplot(data=planets, x='year', kind='count', hue='method',order=range(2005,2015))

tips = sns.load_dataset('tips')
print(tips.head())

# сравнение числовых данных
## Числовые Пары

sns.pairplot(tips)

##тепловая карта
tips_corr = tips[['total_bill', 'tip', 'size']]
sns.heatmap(tips_corr.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1) #
#0-независимые
# 1- прямопорпорциональная
# -1 - обратнопропорциональныя

##диаграмма рассеения

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')

sns.regplot(data=tips, x='total_bill', y='tip')#линейная регрессия
sns.relplot(data=tips, x='total_bill', y='tip', hue='sex')

## простой линейный график

sns.lineplot(data=tips, x='total_bill', y='tip')

##Сводная диаграмма

sns.jointplot( data=tips,x='total_bill', y='tip')

##Сравнение числовых и категориальных данных
#гистограмма
sns.barplot(data=tips, y='total_bill', x='day', hue='sex')

#точки
sns.pointplot(data=tips, y='total_bill', x='day', hue='sex')

#Ящик "с усами"

sns.boxplot(data=tips, y='total_bill', x='day')

#скрипичная диаграмма

sns.violinplot(data=tips, y='total_bill', x='day')

#одномерная диаграмма рассеяния

sns.stripplot(data=tips, y='total_bill', x='day')



plt.show()
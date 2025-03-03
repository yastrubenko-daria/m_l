# 1. сценарий
# 2. командная оболочка IPython
# 3. jupyter

# 1.

# plt.show() запускается один раз
# Figure

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(1,10,100)

fig = plt.figure()
plt.plot(x,np.sin(x))

plt.plot(x,np.cos(x))
plt.show()

# IPython
# %matplotlib
# import matplotlib.pyplot as plt
# plt.plot(...) - открывает окно графика
# plt.draw()
#
# Jupyther
# %matplotlib inline - в блокнот добавляется статическая картинка
# %matplotlib notebook - в блокнот добавляются интерактивные графики

fig.savefig('saved_images.png')

#print(fig.canvas.get_supported_filetypes())


# два способа вывода графиков
# MATLAB - подобный стиль
# в Объектно орентированном стиле

#1.
x=np.linspace(1,10,100)
plt.figure()

plt.subplot(2,1,1)
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))

plt.show()

#2. fig:plt.Figure - контейнер, который содержит все объекты ( СК, тексты, метки), ax:Axes - система координат - прямоугольник, делениея, метки
fig, ax = plt.subplots(2)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

# цвета линий color
# - 'blue'
#- ''rbgcmyk' ->'rg'
# - '0.14' -градация серого от 0 до 1
# - RRGGBB - 'FF00EE'
# - RGB - (1.0, 0.2, 0.3)
# HTML - 'salmon'
#
# Стиль линии linestyle
# - сплошная '-', 'solid'
# - штриховая '--', 'dashed'
# - штрихпунктирная '-.', 'dashdot'
# - пунктирная ':', 'dotted'

x=np.linspace(1,10,100)
fig = plt.figure()
ax = plt.axes()
ax.plot(x, np.sin(x), color='blue')
ax.plot(x, np.sin(x-1), color='g', linestyle='solid')
ax.plot(x, np.sin(x-2), color='0.75',  linestyle='dashed')
ax.plot(x, np.sin(x-3), color='#FF00EE',  linestyle='dashdot')
ax.plot(x, np.sin(x-4), color = (1.0, 0.2, 0.3),  linestyle='dotted')
ax.plot(x, np.sin(x-5), color='salmon',  linestyle=':')
ax.plot(x, np.sin(x-6), '--g') #k- черный

fig, ax= plt.subplots(4)

x=np.linspace(1,10,100)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.sin(x))
ax[2].plot(x, np.sin(x))
ax[3].plot(x, np.sin(x))

ax[1].set_xlim(-2, 12)
ax[1].set_ylim(-1.5,1.5)

ax[2].set_xlim(12, -2)
ax[2].set_ylim(1.5, -1.5)

ax[3].autoscale(tight=True)

plt.subplot(3,1,1)
plt.plot(x, np.sin(x))

plt.title('синус')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.subplot(3,1,2)
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')

plt.legend()

plt.subplot(3,1,3)
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')

plt.title('синус и косинус')
plt.xlabel('x')
plt.axis('equal')# один масштаб у x и y
plt.legend()

plt.subplots_adjust(hspace = 0.5)

x=np.linspace(1,10,30)
plt.plot(x, np.sin(x), 'o', color='green')
plt.plot(x, np.sin(x)+1, '*', color='green')
plt.plot(x, np.sin(x)+2, '^', color='green')
plt.plot(x, np.sin(x)+3, '>', color='green')
plt.plot(x, np.sin(x)+4, 's', color='green')

plt.plot(x, np.sin(x), '--p', markersize=15, linewidth=4, markerfacecolor='white', markeredgecolor = 'gray', markeredgewidth=2)

rng = np.random.default_rng(0)

colors= rng.random(30)
sizes= 30*rng.random(30)
plt.scatter(x, np.sin(x), marker='o', c=colors, s = sizes)
plt.colorbar()

# Еслм точек больше 1000, то plot предпочтительнее изза производительности
# визуализация погрешности

x=np.linspace(1,10,50)
dy=0.4
y= np.sin(x) +dy*np.random.randn(50)

plt.errorbar(x,y,yerr=dy, fmt='.k')
plt.fill_between(x, y-dy, y+dy, color='red', alpha=0.4)
def f(x,y):
    return np.sin(x)**5 +np.cos(20+x*y)*np.cos(x)

x=np.linspace(0,5,50)
y=np.linspace(0,5,40)

X,Y=np.meshgrid(x,y)

Z=f(X,Y)
plt.contour(X,Y,Z, cmap='RdGy')

plt.contourf(X,Y,Z, cmap='RdGy') #залить
c=plt.contour(X,Y,Z, color='red')
plt.clabel(c)# значение циферки
plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian', origin='lower', aspect='equal') # сгладить
plt.colorbar()

plt.show()





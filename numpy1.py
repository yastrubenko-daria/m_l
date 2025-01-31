import numpy as np
# суммирование значений

rng = np.random.default_rng(1)
s=rng.random(50)

print(s)
print(sum(s))
print(np.sum(s))# считает суммы быстрее больших объёмов и умеет считать двумерные и т.д.

a=np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(np.sum(a))
print(np.sum(a, axis=0)) #строка
print(np.sum(a, axis=1))#столбец

print(np.min(a))
print(np.min(a, axis=0))
print(np.min(a, axis=1))

print(a.min())
print(a.min(0))
print(a.min(1))

print(np.nanmin(a))# если есть NaN
print(np.nanmin(a, axis=0))
print(np.nanmin(a, axis=1))

#транслирование (broadcasting)
# набор правил, которые позволяют осуществлять бинарные операции с массивами разных форм и размеров

a= np.array([0,1,2])
b= np.array([5,5,5])

print(a+b)
print(a+5) # "5" транслируется в [5.5.5], то есть она подстраивается под размер массива а

a=np.array([[0,1,2],[3,4,5]])
print(a+5)
print(' ')
a= np.array([0,1,2])
b= np.array([[0],[1],[2]])
print(a+b)

# Правила:
# 1. если размерность массива отличается, то форма массива с меньшей резмерностью дополняется единицей с левой стороны
a= np.array([[0,1,2], [3,4,5]])
b=np.array([5])
print(a.ndim, a.shape)
print(b.ndim, b.shape)
#b(1,) -> (1,1)
# 2.Если формы массивов не совпадают в каком то измерении, то если у массива форма равна 1, то он растягивается до соответствия формы второго
#a        (2,3)
#b(1,) -> (1,1) -> (2,3)
#3. если в каком либо измерении размеры отличаются и ни один из них не равен 1, то генерируется ошибка
a=np.ones((2,3))
b=np.arange(3)
print(a)
print(b)
print(a.ndim, a.shape)
print(b.ndim, b.shape)
#a(2,3)   (2,3)    (2,3)
#b(3,) -> (1,3) -> (2,3)

c=a+b
print(c, c.shape)

a=np.arange(3).reshape((3,1))
b=np.arange(3)
print(a)
print(b)
print(a.ndim, a.shape)
print(b.ndim, b.shape)
#a(3,1)   (3,1) -> (3,3)
#b(3,) -> (1,3) -> (3,3)
c=a+b
print(c, c.shape)

a=np.ones((3,2))
b=np.arange(3)
print(a)
print(b)
print(a.ndim, a.shape)
print(b.ndim, b.shape)
#a(3,2)   (3,2)    (3,3)
#b(3,) -> (1,3) -> (3,3)
#c=a+b
#print(c, c.shape) # ошибка

#Что нужно изменить в примере, чтоб он работал без ошибок
print('------задание1------------')



X = np.array([[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]])
Xmean0=X.mean(0) # среднее значение по строке
print(Xmean0)
Xcenter0 = X-Xmean0
print(Xcenter0)

Xmean1=X.mean(1) # среднее значение по столбцу
print(Xmean1)
Xmean1=Xmean1[:, np.newaxis]
Xcenter1 = X-Xmean1
print(Xcenter1)

x=np.linspace(0,5,50)
y=np.linspace(0,5,50)[:,np.newaxis]

z=np.sin(x)**3 +np.cos(20+y*x)*np.sin(y)

print(z.shape)

import matplotlib.pyplot as plt
plt.imshow(z)
plt.colorbar()
#plt.show()

print('------Сравнение-------')
# Сравнение
x=np.array([1,2,3,4,5])
y=np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(x<3)
print(np.less(x,3))

#False=0
#True=1

print(np.sum(x<3))# количество элементов
print(np.sum(y<4, axis=0))
print(np.sum(y<4, axis=1))
print(np.sum(y<4))

# & | ^ -

#Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9
print('-----задание2------')
print((y>3) & (y<9))
print(np.sum((y>3) & (y<9)))# количество элементов
print(np.sum((y>3) & (y<9), axis=0))
print(np.sum((y>3) & (y<9), axis=1))

#Маски - булевы массивы
x=np.array([1,2,3,4,5])
y=print(x<3)
print(x[x<3])

print(bin(42))
print(bin(59))
print(bin(42&59))

#Векторизация индекса
print('------Векторизация индекса------')
x=np.array([0,1,2,3,4,5,6,7,8,9])
index=[1,5,7]

print(x[index])

index = [[1,5,7],[2,4,8]]
print(x[index])

#Форма результата отражает форму массива индексов, а не форму исходного массива

x=np.arange(12).reshape((3,4))
print(x)
print(x[2])
print(x[2,[2,0,1]])
print(x[1:,[2,0,1]])

x=np.arange(10)
i=np.array([2,1,8,4])
print(x)
x[i] = 999
print(x)

# Сортировка

print('-------Сортировка-----------')
x=[3,2,3,5,2,6,7,3,6,3,2]

print(sorted(x))
print(np.sort(x))
print(x.sort())
#Структурирование массива
print('------структурирование массива------')
data = np.zeros(4, dtype={
    'names':('name', 'age'),
    'formats':('U10', 'i4')
})

print(data.dtype)

name=['name1', 'nae2', 'name3', 'name4']
age=[10,20,30,40]

data['name']=name
data['age']=age

print(data)

print(data['age']>20)
print(data[data['age']>20]['name'])

# массивы записей

data_rec = data.view(np.recarray)
print(data_rec)
print(data_rec[0])
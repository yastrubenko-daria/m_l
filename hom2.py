import numpy as np
#Что нужно изменить в примере, чтоб он работал без ошибок
print('------задание1------------')
#1)заменить a=np.ones((3,2)) на a=np.ones((2,3))

a=np.ones((3,2))
a=a.reshape((2,3))
b=np.arange(3)
c=a+b
print(c, c.shape)
#2)заменить b=np.arange(3) на b=np.arange(3, 1)
a=np.ones((3,2))
b=np.arange(3)
b=b.reshape((3,1))
c=a+b
print(c, c.shape)

#Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9
print('-----задание2------')
y=np.array([[1,2,3,4,5], [6,7,8,9,10]])
print((y>3) & (y<9))
print(np.sum((y>3) & (y<9)))# количество элементов
print(np.sum((y>3) & (y<9), axis=0))
print(np.sum((y>3) & (y<9), axis=1))
import array

import numpy as np
import sys

# Типы данных питона -> динамическая типизация

#x=1
#print(type(x))
#print(sys.getsizeof(x))
#x='hello'
#print(type(x))

#l1=list([])
#print(sys.getsizeof(l1))
#
#l2=list([1,2,3])
#print(sys.getsizeof(l2))
#
#l3=list([1,"2",True])
#print(sys.getsizeof(l3))

#a1=array.array('i',[1,2,3])
#print(sys.getsizeof(a1))
#print(type(a1))

##1. какие ещё существуют коды типов?
##2. Напишите код но с другим типом

#a=np.array([1,2,3,4,5])
#print(type(a), a)
#
#a=np.array([1.23,2,3,4,5]) # приводит к одному типу
#print(type(a), a)
#
#a=np.array([1.23,2,3,4,5], dtype=int) # а я хочу int
#print(type(a), a)
#
#a=np.array([range(i,i+3) for i in [2,4,6]])
#print(type(a), a)

#a =np.zeros(10, dtype=int)
#print(type(a), a)
#print(np.ones((3,5),dtype=float))
#print(np.full((4,5),3.1415))
#print(np.arange(0,20,2))
#print(np.eye(4))

##3. напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапозоне от 0 до 1

##4. напишите код для создания массива с 5 равномерными распределёнными случайными значениями, в диапозоне от 0 до 1

##5. напишите код для создания массива с 5 нормально распределёнными случайными значениями с мат ожиданием =0 и дисперсией 1

##6. напишите код для создания массива с 5 случайными целыми числами в [0,10)


### МАССИВЫ

np.random.seed(1)

#x1=np.random.randint(10, size=3)
#x2=np.random.randint(10, size=(3,2))
#x3=np.random.randint(10, size=(3,2,1))
#print(x1)
#print(x2)
#print(x3)
#
#print(x1.ndim, x1.shape, x1.size)# число размерностей, число каждой размерности, общий размер массива
#print(x2.ndim, x2.shape, x2.size)
#print(x3.ndim, x3.shape, x3.size)

###иНДЕКСЫ (с 0)
#a=np.array([1,2,3,4,5])
#print(a[0])
#print(a[-2])
#a[1]=20
#print(a)
#
#a=np.array([[1,2],[3,4]])
#print(a)
#
#print(a[0,0])
#print(a[-1,-1])
#a[1,0]=100
#print(a)

#a= np.array([1,2,3,4])
#b= np.array([1.0,2,3,4])
#a[0]=10
#a[0]=10.123# при изменении тип не меняется
#print(a,b)

###СРЕЗЫ [s:f:st] [0:shape:1]

#a=np.array([1,2,3,4,5,6])
#b=a[:3] #ссфлка на элемент массива
#print(b)
#b[0]=100
#print(a)
#print(a[0:3:1])
#print(a[:3])
#print(a[3:])
#print(a[1:5])
#print(a[1:-1])
#print(a[1:6:2])
#print(a[1::2])
#print(a[::1])
#print(a[::-1])

##7. написать код для создания срезов массива 3 на 4
# - первые две строки и три солбца
# - первые три строки и второй столбец
# - все строки и столбцы в обратном порядке
# - вторрой столбец
# - третья строка

##8.как сделать срез-копию


#a=np.arange(1,13)
#print(a)
#print(a.reshape(2,6)) #изменение размера
#print(a.reshape(3,4))

##9. newaxis. Продемонстрируйте использование для получения вектора столбца и вектора строки

###ОБЪЕДИНЕНИЕ МАССИВОВ

#x=np.array([1,2,3])
#y=np.array([4,5])
#z=np.array([6])
#
#print(np.concatenate([x,y,z]))

#x=np.array([1,2,3])
#y=np.array([4,5,6])
#
#r1=np.vstack([x,y])
#print(r1)
#
#print(np.hstack([r1,r1]))
##10.Разберитесь как работает метод dstack

##11.Разберитесь как работают методы split, vsplit, hsplit, dsplit

###ВЫЧИСЛЕНИЯ С МАССИВАМИ. ВЕКТОРИЗИРОВАННАЯ ОПЕРАЦИЯ- независимо к каждому элементу

#x=np.arange(10)
#print(x)
#
#print(x*2+1)
#print(np.add(np.multiply(x,2),1))# унивкрсальные функции

# - -  / // ** %

##12.привести пример исспользование всех универсальных функицй

##np.abs, sin, cos, tan, atan, exp, log,...

#x=np.arange(5)
#y=np.empty(5)# пустой
#print(np.multiply(x,10, out=y))
#print(y)
#
#x=np.arange(5)
#y=np.zeros(10)
#print(np.multiply(x,10, out=y[::2]))
#print(y)

x= np.arange(1,5)

print(x)
print(np.add.reduce(x))# сумма
print(np.add.accumulate(x))# записываться на каждом этапе

x=np.arange(1,10)
print(np.add.outer(x,x))# таблица сложения
print(np.multiply.outer(x,x))# таблица умножения
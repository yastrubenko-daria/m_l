import array
import numpy as np
##1. какие ещё существуют коды типов?

# int, float, Unicode character,

##2. Напишите код но с другим типом
a1=array.array('f',[1.23,2,3])
#print(a1)
a2=array.array('i',[True,False,False])
#print(a2)
a3=array.array('u','hello \u2641')
#print(a3)

##3. напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапозоне от 0 до 1
a4=np.arange(0,1,0.2)
#print(a4)
##4. напишите код для создания массива с 5 равномерными распределёнными случайными значениями, в диапозоне от 0 до 1
a5=np.random.rand(5)
#print(a5)
##5. напишите код для создания массива с 5 нормально распределёнными случайными значениями с мат ожиданием =0 и дисперсией 1
a6=np.random.normal(size=5)
#print(a6)
##6. напишите код для создания массива с 5 случайными целыми числами в [0,10)
a7=np.random.randint(0,9,5)
#print(a7)

##7. написать код для создания срезов массива 3 на 4
# - первые две строки и три солбца
# - первые три строки и второй столбец
# - все строки и столбцы в обратном порядке
# - вторрой столбец
# - третья строка

x=np.random.randint(20, size=(3,4))
print(x)
print(x[:2,:3])
print(x[:3, 1])
print(x[::-1,::-1])
print(x[:,1])
print(x[2,:])

##8.как сделать срез-копию
x1=np.random.randint(20, size=10)
x_copy2=x1.copy()
x_copy2[-1]=45
print(x1)

##9. newaxis. Продемонстрируйте использование для получения вектора столбца и вектора строки
y=np.random.randint(20, size=5)
print(y[np.newaxis, :])
print(y[:, np.newaxis])
##10.Разберитесь как работает метод dstack
x=np.array([1,2,3])
y=np.array([4,5,6])
print(np.dstack([x,y]))
##11.Разберитесь как работают методы split, vsplit, hsplit, dsplit
ar=np.arange(16).reshape(4,4)
print(np.array_split(ar,3,1))
print(np.vsplit(ar,2))
print(np.hsplit(ar,2))
d=np.dstack([x,y])
print(np.dsplit(d,2))
print(np.split(ar,2))

##12.привести пример исспользование всех универсальных функицй
arr=np.random.randint(-20,20, size=5)
print(arr)
print(-arr)
print(arr-1)
print(arr/2)
print(arr//2)
print(arr%2)
print(arr**2)
print(abs(arr))
print(np.sqrt(abs(arr)))
print(np.exp(arr))
print(np.log(abs(arr)))
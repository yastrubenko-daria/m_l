import numpy as np
import pandas as pd
# для сосздания Series можно использовать
# 1) списки питона или массивы из numpy
# 2) скалярные значения
# 3) словари

# 1.Привести различные способы создания объектов типа Series
print('--------задание1---------')
mas = [1, 2, 3.2, 4]
obj_1 = pd.Series(mas)
print(obj_1)

mas_2 = np.array([1, 2.8, 3, 4])
obj_2 = pd.Series(mas_2)
print(obj_2)

obj_3 = pd.Series(50.5, index=['при', 'пре', 'пере'])
print(obj_3)

d = {
    'age_1': 51.,
    'age_2': 52,
    'age_3': 53,
}
obj_4 = pd.Series(d)
print(obj_4)


# DataFrame. Способы создания
# 1) через объекты Series
# 2) списки словарей
# 3) словари объектов Series
# 4) двумерный массив NumPy
# 5) структурированный массив

# 2.Привести различные способы создания объектов типа DataFrame
print('--------задание2---------')
d_1 = {
    'people_1': 'Nastya',
    'people_2': 'Jim',
    'people_3': 'Jane',
}
d_2 = {
    'people_1': 25,
    'people_2': 35,
    'people_3': 15,
}
obj_11 = pd.Series(d_1)
obj_12 = pd.Series(d_2)
obj_1 = pd.DataFrame({
    'name': d_1,
    'age': d_2,
})
df_1 = pd.DataFrame(obj_12, columns=['age'])
print(obj_1)
print(df_1)

mas_2 = [{'name': 'Nastya', 'age': 25.}, {'name': 'Jim', 'age': 35.}, {'name': 'Jane', 'age': 15.}]
obj_2 = pd.DataFrame(mas_2)
print(obj_2)

d = {
    'people_1': ['Nastya', 25],
    'people_2': ['Jim', 35],
    'people_3': ['Jane', 15],
}
obj_3 = pd.DataFrame(d)
print(obj_3)

mas_4 = np.array([[1, 5, 7], [3, 6.9, 7], [2, 90, 4]])
obj_4 = pd.DataFrame(mas_4, columns=['a', 'b', 'c'])
print(obj_4)

mas_5 = np.array([(1, 5, 7), (3, 6.9, 7), (2, 90, 4)], dtype=[('a', 'f8'), ('b', 'f8'), ('c', 'f8')])
obj_5 = pd.DataFrame(mas_5)
print(obj_5)

# объедините два объекта Series с неодинаковыми множествами ключей(индексами) так, чтобы вместо Тфт было установлено значение 1

print('-------------задание3-------------')

population_dict = {
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_41': 1004,
    'city_51': 1005,
}

area_dict = {
    'city_1': 9991,
    'city_2': 9992,
    'city_3': 9993,
    'city_42': 9994,
    'city_52': 9995,
}
pop = pd.Series(population_dict)
area = pd.Series(area_dict)

data = pd.DataFrame({
    'pop1': pop,
    'area1': area,
})

data.fillna(1, inplace=True)
print(data)

# переписать пример с транслированием для df так, чтобы вычитание происходило не по строкам а по столбцам
print('------------задание 4-----------------')
rng = np.random.default_rng(1)

A = rng.integers(0, 10, (3, 4))
df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df)

print(df.iloc[:, 0])
print(df.sub(df.iloc[:, 0], axis=0))
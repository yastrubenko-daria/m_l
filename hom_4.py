import pandas as pd
import numpy as np

index =[
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]
index=pd.MultiIndex.from_tuples(index)
pop = pd.Series(population, index=index)

pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)
print(pop_df)
pop_df_1 =pop_df.loc['city_1', 'something']
pop_df_2=pop_df.loc[['city_1', 'city_3'], ['total', 'something']]
pop_df_3=pop_df.loc[['city_1', 'city_3'],  'something']
print(pop_df_1)
print(pop_df_2)
print(pop_df_3)
#1. разобраться


#2. Из получившихся данных выбрать данные по
#-2020 году (для всех столбцов)
#-job_1 (для всех строк)
#- для city_1 и job_2

data = {
    ('city_1', 2010):100,
    ('city_1', 2020): 200,
    ('city_2', 2010): 1001,
    ('city_2', 2020): 2001,
}
s=pd.Series(data)
s.index.names=['city', 'year']

index=pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010,2020]
    ],
    names=['city', 'year']
)
columns=pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1','job_2']
    ],
    names=['worker', 'job']
)
rng = np.random.default_rng(1)
data= rng.random((4,6))
data_df =pd.DataFrame(data, index=index, columns=columns)
print(data_df)
print(data_df.xs(2020, level='year'))
print(data_df.xs('job_1', level='job', axis=1))
print(data_df.xs('job_2', level='job', axis=1).xs('city_1', level='city'))
#3.  Взять за основу DataFrame
index=pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010,2020]
    ],
    names=['city', 'year']
)
columns=pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1','job_2']
    ],
    names=['worker', 'job']
)
rng = np.random.default_rng(1)
data= rng.random((4,6))
df=pd.DataFrame(data, index=index, columns=columns)
print(df)
#выполнить запрос на получение следующих данных
#- все данные по person_1 и person_3
#все данные по первому городу и первым двум person-ам (с использываением срезов)
#приведитее пример ( самостоятельно) с использыванием pd.IndexSlice
print(df.loc[:,['person_1', 'person_3']])
print(df.loc['city_1','person_1':'person_2'])
idx=pd.IndexSlice
print(df.loc[:,idx[:,'job_1']])
#4.привести пример использования inner outer join для Series (на данных предыдущего примера)
ser1=pd.Series({1: 'a', 2:'b', 4:'d'})
ser2=pd.Series({4: 'd', 2:'e', 6:'f'})
ser3=pd.Series({5: '8', 7:'b', 9:'4'})
ser4=pd.Series({7: '0', 9:'0', 8:'f'})
df1=pd.DataFrame({1:ser1,2: ser2})
df2=pd.DataFrame({2:ser3,3: ser4})
print(pd.concat([df1, df2], join='outer'))#по столбцам
print(pd.concat([df1, df2], join='inner'))


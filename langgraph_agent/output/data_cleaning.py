import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('student_habits_performance.csv')

# 处理缺失值
print('处理缺失值前的数据列情况:')
print(df.info())

# 填充或删除缺失值
# 由于 parental_education_level 缺失值较多(91个)，选择删除该列
if df['parental_education_level'].isnull().sum() > 0:
    df = df.dropna(subset=['parental_education_level'])
    print('已删除 parental_education_level 列的缺失值')
else:
    print('parental_education_level 列无缺失值')

# 检查重复数据
print('检查重复数据:')
duplicates = df.duplicated().sum()
print(f'重复数据数量: {duplicates}')
if duplicates > 0:
    df = df.drop_duplicates()
    print('已删除重复数据')
else:
    print('无重复数据')

# 检查异常值
print('检查异常值:')
for col in df.select_dtypes(include=['number']).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'列 {col} 的异常值情况:')
    print(outliers)
    df = df[~(df[col] < lower_bound) | (df[col] > upper_bound)]

# 保存处理后的数据
df.to_csv('student_habits_performance_cleaned.csv', index=False)
print('数据清洗完成，保存到 student_habits_performance_cleaned.csv')
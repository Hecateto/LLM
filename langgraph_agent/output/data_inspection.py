import pandas as pd

df = pd.read_csv('student_habits_performance.csv')
print('列名:', df.columns.tolist())
print('数据类型:\n', df.dtypes.to_string())
print('前几行数据:\n', df.head().to_string())
print('缺失值情况:\n', df.isnull().sum().to_string())
print('异常值检查（数值列范围）:\n', df.describe().to_string())
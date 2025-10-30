import pandas as pd
# Load and check CSV file
try:
    df = pd.read_csv('student_habits_performance.csv')
    print('文件加载成功')
    print('数据预览：
', df.head())
    print('数据信息：
', df.info())
except FileNotFoundError:
    print('错误：文件未找到')
    exit()
except Exception as e:
    print(f'错误：{str(e)}')
    exit()
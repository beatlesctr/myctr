import pandas as pd

data = {
        '性别':['男','女','女','男','男'],
        '姓名':['小明','小红','小芳','大黑','张三'],
        '年龄':[20,21,25,24,29]}
df = pd.DataFrame(data,
               columns=['姓名','性别','年龄','职业'])
print(df)
data = df.loc[[1,4,3]]

print(data)
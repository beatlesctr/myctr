import numpy
import pandas as pd

data = {
        '性别':['男,1;nan,2','女,1','女,1','男,1','男,1'],
        '姓名':['小明','小红','小芳','大黑','张三'],
        '年龄':[20,21,25,24,29]}
df = pd.DataFrame(data,
               columns=['姓名','性别','年龄','职业'])
print(df)
def seq_proc(seq_row, max_seq_len, cell_size):
    cel_str_list = seq_row.split(';') # ['男,1', 'nan,2']
    cel_str_container = [['mynull' for j in range(cell_size)] for i in range(max_seq_len)]
    for i in range(len(cel_str_list)):
        cel_str_container[i] = cel_str_list[i].split(',')
    return cel_str_container


df['性别'] = df['性别'].apply(lambda x : seq_proc(x, 10, 2))

a = [1,2,3,4,5,6]
ret = numpy.array(a).reshape((3, 2))
print(ret)
print(df)

a = [1,2,3]
a_t = tuple(a)
print(a_t)

print('a %d'%(1))
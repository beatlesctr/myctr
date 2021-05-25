import random
n = [i for i in range(100)]
random.shuffle(n)
print(n)

import pandas as pd

df = pd.read_csv(filepath_or_buffer="a.txt", sep=':')
df

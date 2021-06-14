import pandas as pd

import pandas as pd

user = pd.read_table(filepath_or_buffer='./raw/users.dat', sep='::')
movie = pd.read_table(filepath_or_buffer='./raw/movies.dat', sep='::')
def process_genres(genres):
    tmp = genres.split('|')
    while len(tmp) < 3:
        tmp.append('None')
    return '%'.join(tmp[:3])
movie['Genres'] = movie.apply(lambda x: process_genres(x['Genres']), axis=1)
movie[['Genres1', 'Genres2', 'Genres3']] = movie['Genres'].str.split('%', 2, expand=True)
del movie['Genres']
rating = pd.read_table(filepath_or_buffer='./raw/ratings.dat', sep='::')
print(rating.head(10))
result = pd.merge(left=rating, right=user, on='UserID', how='left')
result = pd.merge(left=result, right=movie, on='MovieID', how='left')
result = result.sort_values(by='Timestamp')[lambda x : x['Rating'] != 3]
del result['Timestamp']

result['Rating'] = result.apply(lambda x: 0 if int(x['Rating']) < 3 else 1, axis=1)
sparse_cols = ["UserID","MovieID","Gender","Age","Occupation","Zip-code","Title","Genres1","Genres2","Genres3"]

sparse_feat_space_cfg = list()
for ele in sparse_cols:
    sparse_feat_space_cfg.append(len(set(result[ele])))
print(sparse_feat_space_cfg)

import random
index = [i for i in range(len(result))]
random.shuffle(index)

train_set_size = int(len(result) * 9 / 10)
test_set_size = int(len(result) / 10)
print(train_set_size)
print(test_set_size)
result.reset_index(drop=True, inplace=True)
train_set = result.loc[result.index.intersection(index[:train_set_size])]
test_set = result.loc[result.index.intersection(index[-test_set_size:])]

train_set.to_csv("./raw/train.txt", sep=':', index=False)
test_set.to_csv("./raw/eval.txt", sep=':', index=False)
print('done')
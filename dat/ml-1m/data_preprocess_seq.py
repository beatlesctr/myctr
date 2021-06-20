import pandas as pd
import numpy as np

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

rating_df = pd.DataFrame.from_dict(rating)

def func(dframe):
    max_len = 3
    dframe = dframe.sort_values(by=['Timestamp'], inplace=False)
    dframe = dframe.reset_index(drop=True)
    print(dframe['MoiveID'].values)

    sample_container = list()

    for index in range(len(dframe)):
        moive_id_seq = dframe['MoiveID'].iloc[0:index].values
        moive_id_seq = np.flipud(moive_id_seq[-max_len:])
        seq_item_container = ['mynull' for i in range(max_len)]
        for i in range(len(moive_id_seq)):
            seq_item_container[i] = moive_id_seq[i]
        seq_item_str = '$'.join(seq_item_container)
        cur_item_str = dframe['MoiveID'].iloc[index]
        seq_str_feat = '#'.join([cur_item_str, seq_item_str])
        rating_str = str(dframe['Rating'].iloc[index])
        sample_str = '#'.join([seq_str_feat, rating_str])
        sample_container.append(sample_str)
    ret = ':'.join(sample_container)
    return ret
rating_series = rating_df.groupby(by='UserID')['UserID', 'MoiveID', 'Timestamp', 'Rating'].apply(func)
rating_dframe = pd.DataFrame({
    'UserID': rating_series.index,
    'Sample': rating_series.values
})

rating_df = rating_dframe.set_index(['UserID'])["Sample"].str.split(':', expand=True).stack()\
    .reset_index(drop=True, level=-1).reset_index().rename(columns={0:'Sample'})

rating_df[['MoiveID', 'Seqfeat', 'Rating']] = rating_df['Sample'].str.split('#', 2, expand=True)
del rating_df['Sample']


result = pd.merge(left=rating_df, right=user, on='UserID', how='left')
result = pd.merge(left=rating_df, right=movie, on='MovieID', how='left')


print(result.head(10))


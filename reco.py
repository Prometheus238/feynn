import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

movies = pd.read_csv("feynn/recdata/movies.csv")
ratings = pd.read_csv("feynn/recdata/ratings.csv")

fdataset = ratings.pivot(index='movieId',columns='userId',values='rating')
fdataset.fillna(0, inplace = True)
print(fdataset)

user_voted_count = ratings.groupby('movieId')['rating'].agg('count')
movies_voted_count = ratings.groupby('userId')['rating'].agg('count')

f,ax = plt.subplots(1,1,figsize=(16,4))

plt.scatter(user_voted_count.index,user_voted_count,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

fdataset = fdataset.loc[user_voted_count[user_voted_count > 10].index,:]

f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(movies_voted_count.index,movies_voted_count,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()

fdataset = fdataset.loc[:,movies_voted_count[movies_voted_count > 50].index]

csr_data = csr_matrix(fdataset.values)
fdataset.reset_index(inplace = True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_recommendation(mov_name):
    rec_count = 10
    movie_list = movies[movies['title'].str.contains(mov_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = fdataset[fdataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=rec_count+1)
        rec_movie_idx = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        rec_frame = []
        for x in rec_movie_idx:
            movie_idx = fdataset.iloc[x[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            rec_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance':x[1]})
        df = pd.DataFrame(rec_frame, index = range(1,rec_count+1))
        return df
    else:
        return "Not Found."

print(get_recommendation('Jumanji'))

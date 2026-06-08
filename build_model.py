import pandas as pd
import pickle
import ast
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 Building model...")

# Correct: Pointing to the file inside the folder
movies = pd.read_csv('movies.csv/tmdb_5000_movies.csv')
credits = pd.read_csv('credits.csv/tmdb_5000_credits.csv')

df = movies.merge(credits, on="title")

df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity']]
df.dropna(inplace=True)

# helpers
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

# apply
df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)
df['cast'] = df['cast'].apply(convert_cast)
df['crew'] = df['crew'].apply(fetch_director)

df['overview'] = df['overview'].apply(lambda x: x.split())

def collapse(L):
    return [i.replace(" ", "") for i in L]

df['genres'] = df['genres'].apply(collapse)
df['keywords'] = df['keywords'].apply(collapse)
df['cast'] = df['cast'].apply(collapse)
df['crew'] = df['crew'].apply(collapse)

# tags
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
df['tags'] = df['tags'].apply(lambda x: " ".join(x))

# vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags'])

# save
os.makedirs("models", exist_ok=True)

pickle.dump(df, open('models/df1.pkl', 'wb'))
pickle.dump(vectors, open('models/vectors.pkl', 'wb'))

indices = pd.Series(df.index, index=df['title'].str.lower()).to_dict()
pickle.dump(indices, open('models/indices1.pkl', 'wb'))

print("✅ Model ready!")
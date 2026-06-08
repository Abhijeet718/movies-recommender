import os
import ast
import pickle
import requests
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

API_KEY = "5eded495b89930740bc85bf61f7d1497"

app = FastAPI()
@app.get("/abhijeet")
def abhijeet():
    return {"message": "hello"}

# ==========================================
# SETUP & DATA LOADING
# ==========================================
templates = Jinja2Templates(directory="templates")
templates.env.cache = {}   
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
df = pickle.load(open('models/df1.pkl', 'rb'))
 
indices = pickle.load(open('models/indices1.pkl', 'rb'))
indices = {k.lower(): v for k, v in indices.items()}
 
vectors = pickle.load(open('models/vectors.pkl', 'rb'))

 

# Clean Genres
def clean_genres(x):
    try:
        if isinstance(x, str):
            data = ast.literal_eval(x)
            if isinstance(data, list):
                return [i['name'].lower() for i in data]
        return str(x).lower().replace("|", " ").split()
    except:
        return []

df['genres'] = df['genres'].apply(clean_genres)

# Fix Datatypes
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def fetch_movie_data(movie_title):
    """Fast fetch for the home page (Search & Grid)"""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": movie_title}

    try:
        data = requests.get(url, params=params, timeout=5).json()

        if data.get("results"):
            movie = data["results"][0]
            poster = ""
            if movie.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"]

            rating = movie.get("vote_average", "N/A")
            year = movie.get("release_date", "")[:4]

            return {
                "poster": poster,
                "rating": rating if rating else "N/A",
                "year": year if year else "N/A"
            }
    except Exception as e:
        print("Error fetching:", movie_title, e)

    return {"poster": "", "rating": "N/A", "year": "N/A"}

def fetch_full_movie_details(movie_title):
    """Deep fetch for the movie details page (Cast, Crew, Trailer)"""
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": movie_title}
    search_data = requests.get(search_url, params=params).json()
    
    if not search_data.get("results"):
        return None
        
    movie_id = search_data["results"][0]["id"]
    
    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    details_params = {"api_key": API_KEY, "append_to_response": "credits,videos"}
    details = requests.get(details_url, params=details_params).json()
    
    crew = details.get("credits", {}).get("crew", [])
    director = next((member["name"] for member in crew if member["job"] == "Director"), "Unknown")
    
    cast = details.get("credits", {}).get("cast", [])
    top_cast = [member["name"] for member in cast[:5]]
    
    videos = details.get("videos", {}).get("results", [])
    trailer_key = next((vid["key"] for vid in videos if vid["site"] == "YouTube" and vid["type"] == "Trailer"), None)
    
    return {
        "title": details.get("title"),
        "overview": details.get("overview"),
        "poster": "https://image.tmdb.org/t/p/w500" + details.get("poster_path", ""),
        "backdrop": "https://image.tmdb.org/t/p/w1280" + details.get("backdrop_path", ""),
        "rating": round(details.get("vote_average", 0), 1),
        "year": details.get("release_date", "")[:4],
        "genres": [g["name"] for g in details.get("genres", [])],
        "director": director,
        "cast": top_cast,
        "trailer_key": trailer_key
    }

def recommend(title, n=10):
    """Recommendation Engine Logic"""
    title = title.lower().strip()
    matches = [k for k in indices if title in k]

    if not matches:
        print("❌ Movie not found:", title)
        return None, []

    matched_title = matches[0]
    idx = indices[matched_title]

    # Get searched movie
    actual_movie_title = df.iloc[idx]['title']
    searched_movie_data = fetch_movie_data(actual_movie_title) 
    searched_movie = {
        "title": actual_movie_title,
        "poster": searched_movie_data["poster"],
        "rating": searched_movie_data["rating"],
        "year": searched_movie_data["year"],
        "genres": df.iloc[idx]['genres'][:3]  
    }

    # Get recommendations
    sim_scores = cosine_similarity(
    vectors[idx],
    vectors
).flatten()

    movie_indices = sim_scores.argsort()[::-1][1:n+1]

    results = []
    for movie_idx in movie_indices:
        movie_title = df.iloc[movie_idx]['title']
        movie_data = fetch_movie_data(movie_title)
        results.append({
            "title": movie_title,
            "poster": movie_data["poster"],
            "rating": movie_data["rating"],
            "year": movie_data["year"],
            "genres": df.iloc[movie_idx]['genres'][:2] 
        })

    return searched_movie, results

# ==========================================
# FASTAPI ROUTES
# ==========================================

@app.get("/")
def home():
    return {"status": "working"}

@app.post("/")
def get_recommendations(request: Request, movie: str = Form(...)):
    searched_movie, recommendations = recommend(movie)
    
    message = ""
    if not searched_movie:
        message = "❌ Movie not found. Try another name."

    return templates.TemplateResponse(
        
        "index.html",
        {
            "request": request,
            "searched_movie": searched_movie,        
            "recommendations": recommendations,      
            "message": message
        }
    )

@app.get("/suggest")
def suggest(query: str):
    query = query.lower()
    return [title for title in indices.keys() if query in title][:5]

@app.get("/movie/{title}")
def movie_detail(request: Request, title: str):
    movie_data = fetch_full_movie_details(title)

    return templates.TemplateResponse(
              # 👈 ADD THIS HERE
        "movie.html",
        {"request": request, "title": title, "movie": movie_data}
    )

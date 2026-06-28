import pandas as pd
import faiss
import requests
import numpy as np

df = pd.read_csv('data/netflix_titles.csv')
dim = 768
index = faiss.IndexFlatL2(dim)

def create_representation(row):
    # representation of the dataframe
    t_representation = f"""Type: {row['type']}, 
    Title: {row['title']},
    Director: {row['director']},
    Cast: {row['cast']},
    Released: {row['release_year']},
    Genre: {row['listed_in']},
    Description: {row['description']}"""
    
    return t_representation

df['t_representation'] = df.apply(create_representation, axis=1)

def build_database():
    x = np.zeros((len(df['t_representation']), dim), dtype='float32')

    for n, representation in enumerate(df['t_representation']):
        if n % 200 == 0:
            print(f'Processing {n} of {len(df["t_representation"])}')
        
        response = requests.post('http://localhost:11434/api/embeddings',  
                                json={'model': 'nomic-embed-text', 'prompt': representation})

        if response.status_code != 200:
            print(f"Ollama Error: {response.text}")
            continue

        #Embedding is the vector rep of the text
        embedding = response.json()['embedding']
        x[n] = np.array(embedding)

    index.add(x)

def write_index_to_file():
    faiss.write_index(index, 'index')

def read_index_from_file():
    global index
    index = faiss.read_index('index')

def search_index(query, k=5):
    response = requests.post('http://localhost:11434/api/embeddings',  
                            json={'model': 'nomic-embed-text', 'prompt': query['t_representation']})

    if response.status_code != 200:
        print(f"Ollama Error: {response.text}")
        return []

    embedding = response.json()['embedding']
    xq = np.array([embedding], dtype='float32')
    
    distances, indices = index.search(xq, k)
    results = []
    for i in indices[0]:
        results.append(df.iloc[i]['t_representation'])
    
    return results

def run_recommender():
    while True:
        print("\n--- Movie for Recommendations ---")
        movie_type = input("Enter Type (Movie/TV Show) or 'quit' to exit: ").strip()
        if movie_type.lower() == 'quit':
            print("Goodbye!")
            break

        movie_title = input("Enter the movie or TV show title: ").strip()
        director_name = input("Enter the director's name: ").strip()
        movie_cast = input("Enter the cast members: ").strip()
        release_year = input("Enter the release year: ").strip()
        movie_genre = input("Enter the genre: ").strip()
        movie_description = input("Enter the description: ").strip()

        movie_representation = f"""Type: {movie_type}, 
        Title: {movie_title},
        Director: {director_name},
        Cast: {movie_cast},
        Released: {release_year},
        Genre: {movie_genre},
        Description: {movie_description}"""

        custom_query = {'t_representation': movie_representation}
        
        recommendations = search_index(custom_query, k=5)
        
        print("="*50)
        print("Top Recommendations:")
        print("="*50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"Recommendation {i}:")
            print(rec)
            print("-" * 50)


if __name__ == '__main__':
    
    # First time setup 

    # print("Building the vector database.")
    # build_database()
    # write_index_to_file()
    # print("Database built and saved to 'index'!")


    # After first

    print("Loading database...")
    read_index_from_file()
    print(f"Successfully loaded {index.ntotal} movies into memory!")
    run_recommender()


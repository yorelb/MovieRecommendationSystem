A Python-based movie and TV show recommendation engine powered by LLMs and vector databases. The system reads the metadata and descriptions of movies, maps them into a high-dimensional vector space, and calculates the distance between them to find a match.

This project was built using the Netflix Titles dataset and runs locally.

Features
 - Local AI Embeddings - Uses Ollama to run the nomic-embed-text model directly on machine.
 - Similarity Search: Uses Meta's FAISS to query thousands of vectors in milliseconds.
 - Data Processing: Parses Pandas DataFrames into textual representations (Type, Title, Director, Cast, Release Year, Genre, Description).
 - Interactive CLI: Terminal-based prompt allowing you to search the database.
 - Movie Engine: Allows you to input movie parameters to see what movies match.
 - Persistent Indexing: Saves the compiled vector database locally.

Tech Stack
 - Python 3
 - Pandas & NumPy
 - FAISS (faiss-cpu)
 - Ollama
 - Requests
 - Docker

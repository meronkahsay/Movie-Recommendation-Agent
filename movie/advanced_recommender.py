# advanced_recommender.py

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# Download NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class AdvancedRecommender:
    def __init__(self, csv_file_path):
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"File {csv_file_path} was not found.")
        self.movies = pd.read_csv(csv_file_path)
        self._preprocess()
        self._init_embedding()
        self._build_vector_store()

    def _preprocess(self):
        # Clean Genre
        self.movies['Genre'] = self.movies['Genre'].apply(lambda x: ', '.join(x.split(',')))
        
        # Create rich MetaText for embedding
        self.movies['MetaText'] = self.movies.apply(lambda row:
            f"Title: {row['Series_Title']}\n"
            f"Director: {row['Director']}\n"
            f"Genre: {row['Genre']}\n"
            f"Plot: {row['Overview']}\n"
            f"Stars: {', '.join([str(row[f'Star{i}']) for i in range(1,5) if pd.notna(row[f'Star{i}'])])}\n"
            f"Year: {row['Released_Year']}\n"
            f"IMDB Rating: {row['IMDB_Rating']}\n"
            f"Meta Score: {row.get('Meta_score', 'N/A')}\n"
            f"Votes: {row['No_of_Votes']}", axis=1)

        # Lemmatize for better matching
        lemmatizer = WordNetLemmatizer()
        def lemmatize_text(text):
            if isinstance(text, str):
                words = text.lower().split()
                return ' '.join([lemmatizer.lemmatize(word) for word in words])
            return ""
        self.movies['MetaText'] = self.movies['MetaText'].apply(lemmatize_text)
        self.movies['movie_id'] = self.movies.index.astype(str)

    def _init_embedding(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _build_vector_store(self):
         persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # Check if DB already exists
         if os.path.exists(persist_directory):
               print("Loading existing vector store from disk...")
               self.vector_store = Chroma(
                  persist_directory=persist_directory,
                  embedding_function=self.embeddings
        )
         else:
             print("Building new vector store and saving to disk...")
             texts = self.movies['MetaText'].tolist()
             metadatas = self.movies.to_dict('records')
             ids = self.movies['movie_id'].tolist()

             self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=persist_directory  # ← Save to disk
        )
             self.vector_store.persist()  # Explicitly save

    def _hybrid_score(self, movie_metadata):
        """Boost movies with high ratings and votes"""
        score = 0.0
        rating = movie_metadata.get('IMDB_Rating')
        meta = movie_metadata.get('Meta_score')
        votes = movie_metadata.get('No_of_Votes')

        if pd.notna(rating):
            score += float(rating) * 0.6
        if pd.notna(meta):
            score += float(meta) / 10 * 0.3  # Meta_score is out of 100 → scale to 10
        if pd.notna(votes):
            score += min(float(str(votes).replace(',', '')), 1e6) / 1e5 * 0.1  # cap votes
        return score

    def recommend(self, query: str, top_n=5):
        # Step 1: Semantic search
        results = self.vector_store.similarity_search(query, k=top_n * 20)

        # Step 2: Filter valid results
        filtered = [
            doc for doc in results
            if pd.notna(doc.metadata.get('IMDB_Rating'))
        ]

        # Step 3: Sort by hybrid score
        sorted_results = sorted(
            filtered,
            key=lambda x: self._hybrid_score(x.metadata),
            reverse=True
        )[:top_n]

        # Step 4: Return clean, structured data (Gemini will explain)
        recommendations = []
        for doc in sorted_results:
            meta = doc.metadata
            recommendations.append({
                "title": meta.get("Series_Title", "Unknown"),
                "year": int(meta.get("Released_Year", 0)),
                "genre": meta.get("Genre", "Unknown"),
                "director": meta.get("Director", "Unknown"),
                "stars": ', '.join([
                    str(meta[f"Star{i}"]) for i in range(1, 5)
                    if f"Star{i}" in meta and pd.notna(meta[f"Star{i}"])
                ]),
                "imdb_rating": float(meta["IMDB_Rating"]),
                "meta_score": meta.get("Meta_score", "N/A"),
                "overview": meta.get("Overview", "N/A"),
                "match_confidence": round(doc.score, 3) if hasattr(doc, 'score') else "N/A"
            })
        return recommendations
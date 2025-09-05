# movie/agent.py

import os
from .advanced_recommender import AdvancedRecommender
from google.adk.agents import Agent

# Set up paths
current_dir = os.path.dirname(__file__)
csv_file_path = os.path.join(current_dir, "imdb_top_1000.csv")

# Initialize recommender
recommender = AdvancedRecommender(csv_file_path)

# Define tool
def recommend(query: str):
    """Recommend movies based on query. Returns structured list for Gemini to explain."""
    try:
        return recommender.recommend(query, top_n=5)
    except Exception as e:
        return [{"error": f"Error fetching recommendations: {str(e)}"}]


# Improved instruction with safe string formatting
instruction = (
    "You are CineBot, a friendly and expert movie recommender. Your job is to help users discover great films.\n\n"
    "RULES:\n"
    "1. ALWAYS use the 'recommend' tool when the user asks for movie suggestions.\n"
    "2. NEVER make up movie details — only use tool results.\n"
    "3. Summarize 3–5 top movies clearly and conversationally.\n"
    "4. For each movie, mention: title, year, genre, director, and why it fits the request.\n"
    "5. Highlight standout features: iconic actors, awards, or cultural impact.\n"
    "6. Keep tone warm and engaging — like a movie-loving friend.\n"
    "7. If no movies match, say so honestly and suggest refining the query.\n\n"
    "EXAMPLE RESPONSE:\n"
    "I've found some great options for you! 'Back to the Future' (1985) is a classic sci-fi comedy directed by Robert Zemeckis, "
    "starring Michael J. Fox. It's funny, inventive, and has stood the test of time with an 8.5/10 rating. "
    "If you like action-packed humor, 'Deadpool' (2016) is another top pick, known for its sharp wit and superhero satire."
)

# Create agent
agent = Agent(
    name="movie_recommender_agent",
    model="gemini-1.5-flash",
    description="A smart movie recommender that combines semantic search with AI explanations.",
    instruction=instruction,
    tools=[recommend],
)

# Required by ADK
root_agent = agent
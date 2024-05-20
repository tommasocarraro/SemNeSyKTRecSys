import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LAST_FM_API_KEY = os.getenv("LAST_FM_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

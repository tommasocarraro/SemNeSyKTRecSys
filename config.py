import os
import sys

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    print("Missing Google API key", file=sys.stderr)
    exit(1)

LAST_FM_API_KEY = os.getenv("LAST_FM_API_KEY")
if LAST_FM_API_KEY is None:
    print("Missing last.fm API key", file=sys.stderr)
    exit(1)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if TMDB_API_KEY is None:
    print("Missing The Movie Database API key", file=sys.stderr)
    exit(1)

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    logger.error("Missing Google API key")
    exit(1)

LAST_FM_API_KEY = os.getenv("LAST_FM_API_KEY")
if LAST_FM_API_KEY is None:
    logger.error("Missing last.fm API key")
    exit(1)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if TMDB_API_KEY is None:
    logger.error("Missing The Movie Database API key")
    exit(1)

PSQL_CONN_STRING = os.getenv("PSQL_CONN_STRING")
if PSQL_CONN_STRING is None:
    logger.error("Missing PostgreSQL connection string")
    exit(1)

PSQL_CONN_STRING_SANS_DB = os.getenv("PSQL_CONN_STRING_SANS_DB")
if PSQL_CONN_STRING is None:
    logger.error("Missing PostgreSQL connection without DB string")
    exit(1)

NEO4J_URI = os.getenv("NEO4J_URI")
if NEO4J_URI is None:
    logger.error("Missing Neo4J URI")
    exit(1)

NEO4J_DBNAME = os.getenv("NEO4J_DBNAME")
if NEO4J_DBNAME is None:
    logger.error("Missing Neo4J database name")
    exit(1)

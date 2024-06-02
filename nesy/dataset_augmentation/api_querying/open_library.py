from psycopg import connect
from psycopg_pool import ConnectionPool

from config import PSQL_CONN_STRING
from nesy.dataset_augmentation import state
from loguru import logger


def fuzzy_search_titles(titles: list[str], pool: ConnectionPool) -> list[list[tuple]]:
    """
    Performs an asynchronous fuzzy search against the PostgreSQL database through the provided connection pool
    Args:
        titles: The title to search for
        pool: The connection pool to use

    Returns:
        The query output
    """
    # raising an exception since setting this at runtime during queries scheduling would require a lock in order
    # to avoid a race condition, which would decrease performance. Do it beforehand instead
    if not state.SET_SIM_THRESHOLD:
        raise RuntimeError(
            "You forgot to set the similarity threshold. Invoke set_sim_threshold"
        )
    logger.info(f"Performing {len(titles)} fuzzy searches against Open Library")
    query = """
        SELECT title, authors, year
        FROM edition_authors_materialized_view
        WHERE title %% %(title)s
        ORDER BY (title <-> %(title)s)
        LIMIT 5"""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                query, [{"title": title} for title in titles], returning=True
            )
            res = []
            while True:
                res.append(cur.fetchall())
                if not cur.nextset():
                    break
    return res


def set_sim_threshold(thresh: float) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold

    Returns:
        None
    """
    if not state.SET_SIM_THRESHOLD:
        conn = connect(PSQL_CONN_STRING)
        # SET commands cannot be used with parameters, so I'm building the query string manually
        cur = conn.execute(f"SET pg_trgm.similarity_threshold = {thresh}")
        conn.commit()
        cur.close()
        conn.close()
        state.SET_SIM_THRESHOLD = True

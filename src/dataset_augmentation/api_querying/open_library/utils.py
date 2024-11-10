from psycopg import AsyncCursor


async def set_sim_threshold(thresh: float, cur: AsyncCursor) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold
        cur: The PostgreSQL connection cursor

    Returns:
        None
    """
    await cur.execute(f"SET pg_trgm.similarity_threshold = {thresh}")

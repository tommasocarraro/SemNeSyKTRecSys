from asyncpg import Pool


async def set_sim_threshold(thresh: float, psql_pool: Pool) -> None:
    """
    Sets the similarity threshold to be used when performing fuzzy queries. Needs to be set each time the
    database is spun up
    Args:
        thresh: The similarity threshold
        psql_pool: The PostgreSQL connection

    Returns:
        None
    """
    async with psql_pool.acquire() as conn:
        await conn.execute(f"SET pg_trgm.similarity_threshold = {thresh}")

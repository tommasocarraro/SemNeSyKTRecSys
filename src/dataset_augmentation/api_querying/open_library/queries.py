query_title_authors = """
    WITH author_names AS (
        SELECT unnest(ARRAY[%(authors)s]) AS author_name
    )
    SELECT MIN(sub.year) AS year
    FROM (
        SELECT
           ea.year,
           ea.title_query <-> LOWER(%(title)s) AS title_distance,
           SUM(ea.name_query <-> an.author_name) OVER () AS author_distance
        FROM editions_authors ea
        JOIN author_names an ON ea.name_query %% an.author_name
        WHERE ea.title_query %% LOWER(%(title)s)
        ORDER BY title_distance, author_distance
    ) sub
"""


query_title_year = """
    WITH cte AS (
        SELECT
            key,
            title_query,
            year,
            string_agg(DISTINCT ea.name, ', ') AS person,
            ea.title_query <-> LOWER(%(title)s) AS distance
        FROM editions_authors ea
        WHERE
            ea.title_query % LOWER(%(title)s)
            AND ea.year = %(year)s
        GROUP BY ea.key, ea.title_query, ea.year, distance
    )
    , cte2 AS (
        SELECT
            title_query,
            year,
            person,
            distance
        FROM cte
        GROUP BY title_query, year, person, distance
        ORDER BY distance
    )
    SELECT person
    FROM cte2
    GROUP BY person
    ORDER BY COUNT(person) DESC
    LIMIT 1;
"""


query_title = """
    SELECT
        person,
        year
    FROM get_book_info_by_title(LOWER(%(title)s))
"""

query_isbn = """
    SELECT
        title,
        person,
        year
    FROM get_book_info_by_isbn(%(isbn)s)
"""

from typing import Any


def make_title_authors_query(how_many_authors: int) -> str:
    from_clauses = ""
    where_clauses = ""
    group_by_clauses = ""
    authors_select = ""
    distance_select = ""

    for i in range(how_many_authors):
        if from_clauses == "":
            from_clauses = f"combined_materialized_view c{i}"
        else:
            from_clauses += (
                f" JOIN combined_materialized_view c{i} ON c0.key = c{i}.key"
            )
        if where_clauses == "":
            where_clauses = (
                f"c{i}.title % LOWER($2) AND c{i}.author_name % LOWER(${i+3})"
            )
        else:
            where_clauses += f" AND c{i}.author_name % LOWER(${i+3})"
        if group_by_clauses == "":
            group_by_clauses = f"c{i}.title, c{i}.author_name"
        else:
            group_by_clauses += f", c{i}.author_name"
        if authors_select == "":
            authors_select = f"c{i}.author_name"
        else:
            authors_select += f" || ', ' || c{i}.author_name"
        if i == how_many_authors - 1:
            authors_select += " AS authors"

        if distance_select == "":
            distance_select = (
                f"(c{i}.title <-> LOWER($2)) + (c{i}.author_name <-> LOWER(${i+3}))"
            )
        else:
            distance_select += f" + (c{i}.author_name <-> LOWER(${i+3})"
        if i == how_many_authors - 1:
            where_clauses += f" AND {distance_select} < 0.5"
            distance_select += " AS distance"
    select_clauses = f"$1 as query_index, c0.title, {authors_select}, MIN(c0.year) AS year, {distance_select}"

    query = f"""
        SELECT
            {select_clauses}
        FROM
            {from_clauses}
        WHERE
            {where_clauses}
        GROUP BY {group_by_clauses}
        ORDER BY distance
        LIMIT 10
    """

    return query


def make_title_year_query() -> str:
    query = f"""
        SELECT
            $1 AS query_index,
            title,
            STRING_AGG(author_name, ', ') as authors,
            year,
            (title <-> LOWER($2)) as distance
        FROM combined_materialized_view
        WHERE
            title % LOWER($2)
            AND title <-> LOWER($2) < 0.5
        AND year ilike $3
        GROUP BY title, year
        ORDER BY distance
        LIMIT 10
    """

    return query


def make_title_query() -> str:
    query = f"""
    SELECT
        $1 AS query_index,
        title,
        MIN(year) as year,
        STRING_AGG(DISTINCT author_name, ', ') as authors,
        title <-> LOWER($2) as distance
    FROM combined_materialized_view
    WHERE 
        title % LOWER($2)
        AND title <-> LOWER($2) < 0.5
    GROUP BY key, title
    ORDER BY distance
    LIMIT 10
    """

    return query


def make_query(params_dict: dict[str, Any]):
    kind = params_dict["kind"]

    if kind == "titles":
        make_query_fn = make_title_query
        args = []
    elif kind == "titles_authors":
        make_query_fn = make_title_authors_query
        args = [len(params_dict["authors"])]
    elif kind == "titles_year":
        make_query_fn = make_title_year_query
        args = []
    else:
        raise ValueError(f"kind {kind} not supported")

    return make_query_fn(*args)

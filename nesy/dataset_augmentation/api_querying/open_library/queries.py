from typing import Any


def make_title_authors_query(params_dict: dict[str, Any]) -> str:
    title = params_dict["title"]
    authors = params_dict["authors"]
    idx = params_dict["idx"]

    from_clauses = ""
    where_clauses = ""
    group_by_clauses = ""
    authors_select = ""
    distance_select = ""

    for i, author in enumerate(authors):
        if from_clauses == "":
            from_clauses = f"combined_materialized_view c{i}"
        else:
            from_clauses += (
                f" JOIN combined_materialized_view c{i} ON c0.key = c{i}.key"
            )
        if where_clauses == "":
            where_clauses = f"c{i}.title % LOWER('{title}') AND c{i}.author_name % LOWER('{author}')"
        else:
            where_clauses += f" AND c{i}.author_name % LOWER('{author}')"
        if group_by_clauses == "":
            group_by_clauses = f"c{i}.title, c{i}.author_name"
        else:
            group_by_clauses += f", c{i}.author_name"
        if authors_select == "":
            authors_select = f"c{i}.author_name"
        else:
            authors_select += f" || ', ' || c{i}.author_name"
        if i == len(authors) - 1:
            authors_select += " AS authors"

        if distance_select == "":
            distance_select = f"(c{i}.title <-> LOWER('{title}')) + (c{i}.author_name <-> LOWER('{author}'))"
        else:
            distance_select += f" + (c{i}.author_name <-> LOWER('{author}')"
        if i == len(authors) - 1:
            where_clauses += f" AND {distance_select} < 0.5"
            distance_select += " AS distance"
    select_clauses = f"{idx} as query_index, c0.title, {authors_select}, MIN(c0.year) AS year, {distance_select}"

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


def make_title_year_query(params_dict: dict[str, Any]) -> str:
    title = params_dict["title"]
    year = params_dict["year"]
    idx = params_dict["idx"]

    query = f"""
        SELECT
            {idx} AS query_index,
            title,
            STRING_AGG(author_name, ', ') as authors,
            year,
            (title <-> LOWER('{title}')) as distance
        FROM combined_materialized_view
        WHERE
            title % LOWER('{title}')
            AND title <-> LOWER(%(title)s) < 0.5
        AND year ilike '{year}'
        GROUP BY title, year
        ORDER BY distance
        LIMIT 10
    """

    return query


def make_title_query(params_dict: dict[str, Any]) -> str:
    title = params_dict["title"]
    idx = params_dict["idx"]

    query = f"""
    SELECT
        {idx} AS query_index,
        title,
        MIN(year) as year,
        STRING_AGG(DISTINCT author_name, ', ') as authors,
        title <-> LOWER({title}) as distance
    FROM combined_materialized_view
    WHERE 
        title % LOWER({title})
        AND title <-> LOWER({title}) < 0.5
    GROUP BY key, title
    ORDER BY distance
    LIMIT 10
    """

    return query


def make_query(params_dict: dict[str, Any]):
    kind = params_dict["kind"]

    if kind == "titles":
        make_query_fn = make_title_query
    elif kind == "titles_authors":
        make_query_fn = make_title_authors_query
    elif kind == "titles_authors":
        make_query_fn = make_title_year_query
    else:
        raise ValueError(f"kind {kind} not supported")

    return make_query_fn(params_dict)

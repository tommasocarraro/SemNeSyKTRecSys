CREATE MATERIALIZED VIEW editions_unnested_isbns AS
SELECT
    key,
    title,
    authors,
    year,
    works,
    UNNEST(isbns) AS isbn
FROM editions;

CREATE MATERIALIZED VIEW editions_authors AS
SELECT
    e.key,
    e.title_query,
    e.year,
    a.name,
    a.name_query
FROM editions e
JOIN authors a ON a.key = ANY(e.authors);
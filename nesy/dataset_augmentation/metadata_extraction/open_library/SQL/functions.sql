CREATE FUNCTION get_book_info_by_isbn(isbn_param TEXT)
RETURNS TABLE (
    book_title VARCHAR(200),
    book_year CHAR(4),
    author_names TEXT[]
)
AS $$
DECLARE
    works_keys TEXT[];
    edition_title VARCHAR(200);
    work_title VARCHAR(200);
    min_year CHAR(4);
    author_keys TEXT[];
BEGIN
    -- Query by ISBN
    SELECT
        title,
        works,
        year
    INTO
        edition_title,
        works_keys,
        book_year
    FROM editions_unnested_isbns
    WHERE isbn = isbn_param;

    -- If works is in result from editions_unnested_isbns, retrieve work info and earliest year
    IF works_keys IS NOT NULL THEN
        SELECT title, authors INTO work_title, author_keys
        FROM works
        WHERE key = ANY(works_keys);

        WITH year_cte AS (
            SELECT year
            FROM editions
            WHERE works && works_keys
        )
        SELECT MIN(year) INTO min_year
        FROM year_cte;

        book_year := min_year;
    END IF;

    -- Retrieve author names
    RETURN QUERY
    SELECT
        COALESCE(edition_title, t.book_title2) AS book_title,
        t.book_year,
        ARRAY_AGG(a.name) AS author_names
    FROM (
        SELECT
            CASE WHEN work_title IS NULL THEN edition_title ELSE work_title END AS book_title2,
            book_year,
            a.key
        FROM authors a
        WHERE key = ANY(COALESCE(author_keys, '{}'::TEXT[]))
    ) t
    CROSS JOIN LATERAL UNNEST(author_keys) WITH ORDINALITY AS u(key, ord)
    LEFT JOIN authors a ON a.key = u.key
    GROUP BY t.book_title2, t.book_year;
END;
$$ LANGUAGE plpgsql;

CREATE FUNCTION get_book_info_by_title(title_param VARCHAR(200))
RETURNS TABLE (
    author_names TEXT,
    book_year CHAR(4)
)
AS $$
DECLARE
    authors_var TEXT[];
    author_names_var TEXT;
    year_var CHAR(4);
BEGIN
    WITH distinct_title_authors AS (
    SELECT DISTINCT ON (title_query, authors)
        authors,
        title_query <-> LOWER(title_param) AS distance,
        COUNT(*) OVER (PARTITION BY title_query, authors) AS dist_count
    FROM editions
    WHERE title_query % LOWER(title_param)
    )
    SELECT authors
    INTO authors_var
    FROM distinct_title_authors
    ORDER BY distance, dist_count DESC
    LIMIT 1;

    SELECT MIN(year)
    INTO year_var
    FROM editions
    WHERE
        title_query % LOWER(title_param)
        AND authors = authors_var
    GROUP BY authors;

    -- Query for author names
    SELECT STRING_AGG(name, ', ')
    INTO author_names_var
    FROM authors
    WHERE key = ANY(authors_var);

    RETURN QUERY
    SELECT
        author_names_var AS author_names,
        year_var AS book_year;
END;
$$ LANGUAGE plpgsql;
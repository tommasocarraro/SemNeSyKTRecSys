CREATE
FUNCTION get_book_info_by_isbn(isbn_param TEXT)
    RETURNS TABLE
            (
                title  VARCHAR(200),
                person TEXT[],
                year   CHAR(4)
            )
AS
$$
DECLARE
    work_keys_var     TEXT[];
edition_title_var VARCHAR(200);
work_title_var    VARCHAR(200);
min_year_var      CHAR(4);
author_keys_var   TEXT[];
year_var          CHAR(4);
BEGIN
-- Query by ISBN
SELECT eui.title,
       eui.works,
       eui.year INTO
        edition_title_var,
        work_keys_var,
        year_var
FROM editions_unnested_isbns eui
WHERE eui.isbn = isbn_param;

-- If works is in result from editions_unnested_isbns, retrieve work info and earliest year
IF work_keys_var IS NOT NULL THEN
SELECT w.title,
       w.authors INTO work_title_var, author_keys_var
FROM works w
WHERE w.key = ANY (work_keys_var);

WITH year_cte AS (SELECT e.year
                  FROM editions e
                  WHERE e.works && work_keys_var)
SELECT MIN(yc.year) INTO min_year_var
FROM year_cte yc;
END IF;

-- Retrieve author names
RETURN QUERY
SELECT COALESCE(edition_title_var, t.book_title2) AS book_title,
       ARRAY_AGG(a.name)                          AS author_names,
       t.year
FROM (SELECT CASE WHEN work_title_var IS NULL THEN edition_title_var ELSE work_title_var END AS book_title2,
             min_year_var                                                                    AS year,
             a.key
      FROM authors a
      WHERE key = ANY(COALESCE(author_keys_var, '{}'::TEXT[]))) t
         CROSS JOIN LATERAL UNNEST(author_keys_var)
WITH ORDINALITY AS u(key, ord)
    LEFT JOIN authors a
ON a.key = u.key
GROUP BY t.book_title2, t.year;
END;
$$ LANGUAGE plpgsql;

CREATE
FUNCTION get_book_info_by_title(title_param VARCHAR(200))
    RETURNS TABLE
            (
                person TEXT,
                year   CHAR(4)
            )
AS
$$
DECLARE
    author_keys_var  TEXT[];
author_names_var TEXT;
year_var         CHAR(4);
BEGIN WITH distinct_title_authors AS (SELECT DISTINCT ON (e.title_query, e.authors) e.authors,
                                                                                  e.title_query <-> LOWER(title_param)                  AS distance,
                                                                                  COUNT(*) OVER (PARTITION BY e.title_query, e.authors) AS dist_count
                                    FROM editions e
                                    WHERE e.title_query % LOWER(title_param)
                                      AND e.authors IS NOT NULL)
SELECT dta.authors INTO author_keys_var
FROM distinct_title_authors dta
ORDER BY dta.distance, dta.dist_count DESC
LIMIT 1;

SELECT MIN(e.year) INTO year_var
FROM editions e
WHERE e.title_query % LOWER(title_param)
  AND e.authors = author_keys_var
GROUP BY authors;

-- Query for author names
SELECT STRING_AGG(a.name, ', ') INTO author_names_var
FROM authors a
WHERE a.key = ANY (author_keys_var);

RETURN QUERY
SELECT author_names_var AS person,
       year_var         AS year;
END;
$$ LANGUAGE plpgsql;
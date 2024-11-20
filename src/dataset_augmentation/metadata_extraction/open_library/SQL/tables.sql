CREATE TABLE authors
(
    key        TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    name_query TEXT NOT NULL
);

CREATE TABLE editions
(
    key         TEXT PRIMARY KEY,
    title       VARCHAR(200),
    title_query VARCHAR(200),
    authors     TEXT [],
    year        CHAR(4),
    works       TEXT [],
    isbns       TEXT []
);

CREATE TABLE works
(
    key     TEXT PRIMARY KEY,
    title   VARCHAR(200),
    authors TEXT []
);
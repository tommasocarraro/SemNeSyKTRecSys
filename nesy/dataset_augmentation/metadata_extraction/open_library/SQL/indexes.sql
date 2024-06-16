CREATE INDEX ON editions USING GIN (title_query gin_trgm_ops);
CREATE INDEX ON editions USING GIN (works);

CREATE INDEX ON editions_authors USING GIN (name_query gin_trgm_ops);
CREATE INDEX ON editions_authors USING GIN (title_query gin_trgm_ops);
CREATE INDEX ON editions_authors (year);

CREATE INDEX ON editions_unnested_isbns (isbn);
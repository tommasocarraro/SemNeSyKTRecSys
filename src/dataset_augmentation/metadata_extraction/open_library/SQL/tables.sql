CREATE TABLE authors(
  key TEXT PRIMARY KEY,
  name text NOT NULL,
  name_query text NOT NULL
);

CREATE TABLE editions(
  key TEXT PRIMARY KEY,
  title varchar(200),
  title_query varchar(200),
  authors text[],
  year char(4),
  works text[],
  isbns text[]
);

CREATE TABLE works(
  key TEXT PRIMARY KEY,
  title varchar(200),
  authors text[]
);


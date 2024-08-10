import sqlite3


class PathsDatabase:
    """
    This class uses an SQLite3 database to store paths between pairs of items from specific domains.
    """

    def __init__(self, db_path: str):
        # initialize the connection the database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # create the entities table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL
            )
            """
        )

        # create an index on the entities table using the name column
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)"
        )

        # create the paths table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                start_id INTEGER NOT NULL,
                end_id INTEGER NOT NULL,
                domain TEXT NOT NULL,
                FOREIGN KEY (start_id) REFERENCES entities(id),
                FOREIGN KEY (end_id) REFERENCES entities(id)
            )
            """
        )
        self.conn.commit()

    def _get_or_create_entity(self, name: str, domain: str) -> int:
        """
        Returns the ID of the entity with the given name in the domain specified.
        If it does not exist, it creates it and then returns it.
        Args:
            name: name of the entity.
            domain: domain of the entity.
        """
        self.cursor.execute(
            """
            SELECT id
            FROM entities
            WHERE name = ? AND domain = ?
            """,
            (name, domain),
        )
        result = self.cursor.fetchone()

        # if the entity doesn't exist, insert it into the table
        if result is None:
            self.cursor.execute(
                """
                INSERT INTO entities (name, domain) VALUES (?, ?)
                """,
                (name, domain),
            )
            self.conn.commit()
            return self.cursor.lastrowid
        return result[0]

    def insert_paths(self, start: str, end: str, paths: list[str], domain: str) -> None:
        """
        Inserts the given paths into the database using a transaction.
        Args:
            start: name of the start item
            end: name of the end item
            paths: list of paths in string format
            domain: domain of the path

        Returns: None
        """
        try:
            self.conn.execute("BEGIN TRANSACTION")

            # get the IDs of the given start and end entities
            start_id = self._get_or_create_entity(start, domain)
            end_id = self._get_or_create_entity(end, domain)

            # prepare the data for optimized query execution
            paths_data = [(path, start_id, end_id, domain) for path in paths]

            self.cursor.executemany(
                """
                INSERT INTO paths (path, start_id, end_id, domain)
                VALUES (?, ?, ?, ?)
                """,
                paths_data,
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def insert_empty_path(self, start: str, end: str, domain: str) -> None:
        """
        Inserts the start and end items into the entities table without any path between them.
        Args:
            start: name of the start item
            end: name of the end item
            domain: domain of the entities

        Returns: None
        """
        self._get_or_create_entity(start, domain)
        self._get_or_create_entity(end, domain)

    def get_paths_by_start(self, start: str, domain: str) -> list[str]:
        """
        Returns all paths starting from start in the given domain
        Args:
            start: name of the start item
            domain: domain of the paths
        """
        self.cursor.execute(
            """
            SELECT p.path
            FROM paths p
            JOIN entities s ON p.start_id = s.id
            WHERE s.name = ? AND s.domain = ? AND p.domain = ?
            """,
            (start, domain, domain),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def get_paths_by_end(self, end: str, domain: str) -> list[str]:
        """
        Returns all paths ending with end in the given domain
        Args:
            end: name of the end item
            domain: domain of the paths
        """
        self.cursor.execute(
            """
            SELECT p.path
            FROM paths p
            JOIN entities e ON p.end_id = e.id
            WHERE e.name = ? AND e.domain = ? AND p.domain = ?
            """,
            (end, domain, domain),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def get_paths_by_start_and_end(
        self, start: str, end: str, domain: str
    ) -> list[str]:
        """
        Returns all paths starting from start and ending with end in the given domain
        Args:
            start: name of the start item
            end: name of the end item
            domain: domain of the paths
        """
        self.cursor.execute(
            """
            SELECT p.path
            FROM paths p
            JOIN entities s ON p.end_id = s.id
            JOIN entities e ON p.end_id = e.id
            WHERE s.name = ? AND e.name = ? AND s.domain = ? AND e.domain = ? AND p.domain = ?
            """,
            (start, end, domain, domain, domain),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def check_paths_computed(self, start: str, end: str, domain: str) -> bool:
        """
        Checks whether the paths between start and end have already been computed
        Args:
            start: name of the start item
            end: name of the end item
            domain: domain of the paths

        Returns: True if the paths between start and end have already been computed
        """
        self.cursor.execute(
            """
            SELECT COUNT(*)
            FROM entities
            WHERE name IN (?, ?) AND domain = ?
            """,
            (start, end, domain),
        )
        return self.cursor.fetchone()[0] != 0

    def check_db_empty(self, domain: str) -> bool:
        """
        Checks whether the database is empty for the given domain
        Args:
            domain: domain of the paths

        Returns: True if the database is empty for the given domain
        """
        self.cursor.execute(
            """
            SELECT COUNT(*)
            FROM entities
            WHERE domain = ?
            """,
            (domain,),
        )
        return self.cursor.fetchone()[0] != 0

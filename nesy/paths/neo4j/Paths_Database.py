import sqlite3


class PathsDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            """
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)"
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                start_id INTEGER NOT NULL,
                end_id INTEGER NOT NULL,
                FOREIGN KEY (start_id) REFERENCES entities(id),
                FOREIGN KEY (end_id) REFERENCES entities(id)
            )
            """
        )
        self.conn.commit()

    def _get_or_create_entity(self, name: str):
        self.cursor.execute(
            """
            SELECT id
            FROM entities
            WHERE name = ?
            """,
            (name,),
        )
        result = self.cursor.fetchone()
        if result is None:
            self.cursor.execute(
                """
                INSERT INTO entities (name) VALUES (?)
                """,
                (name,),
            )
            self.conn.commit()
            return self.cursor.lastrowid
        return result[0]

    def insert_path(self, path: str) -> None:
        parts = path.split("->")
        start = parts[0].strip()
        end = parts[-1].strip()

        start_id = self._get_or_create_entity(start)
        end_id = self._get_or_create_entity(end)

        self.cursor.execute(
            """
                INSERT INTO paths (path, start_id, end_id)
                VALUES (?, ?, ?)
            """,
            (path, start_id, end_id),
        )
        self.conn.commit()

    def get_paths_by_start(self, start: str) -> list[str]:
        self.cursor.execute(
            """
            SELECT p.path
            FROM paths p
            JOIN entities s ON p.start_id = s.id
            WHERE s.name = ?
            """,
            (start,),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def get_paths_by_end(self, end: str) -> list[str]:
        self.cursor.execute(
            """
            SELECT p.path
            FROM paths p
            JOIN entities e ON p.end_id = e.id
            WHERE e.name = ?
            """,
            (end,),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def get_paths_by_start_and_end(self, start: str, end: str) -> list[str]:
        self.cursor.execute(
            """
            SELECT p.path
            FROM paths p
            JOIN entities s ON p.end_id = s.id
            JOIN entities e ON p.end_id = e.id
            WHERE s.name = ? AND e.name = ?
            """,
            (
                start,
                end,
            ),
        )
        return [row[0] for row in self.cursor.fetchall()]

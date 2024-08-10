import sqlite3


class Queue:
    """
    This class uses an SQLite database in order to implement a queue.
    Items are not removed from the queue with the dequeue operation for resiliency reasons. Use the manual remove method.
    """

    def __init__(self, queue_path: str):
        # initialize the connection to the database
        self.conn = sqlite3.connect(queue_path)
        self.cursor = self.conn.cursor()

        # create the queue table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL,
            dispatched BOOLEAN NOT NULL
            )
            """
        )

        # reset the dispatched flag in order from previous runs
        self.cursor.execute(
            """
            UPDATE queue
            SET dispatched = FALSE
            """
        )
        self.conn.commit()

    def enqueue(self, data: str) -> None:
        """
        Inserts an item at the end of the queue
        Args:
            data: string containing the data to be inserted

        Returns: None
        """
        self.cursor.execute(
            """
                INSERT INTO queue(data, dispatched)
                VALUES (?, FALSE)
                """,
            (data,),
        )
        self.conn.commit()

    def dequeue(self) -> tuple[int, str]:
        """
        Returns the first non-dispatched item from the queue
        """
        self.cursor.execute(
            """
            SELECT id, data
            FROM queue
            WHERE dispatched = FALSE
            ORDER BY id ASC
            LIMIT 1
            """
        )
        item = self.cursor.fetchone()
        if item:
            # set the dispatched flag to true
            self.cursor.execute(
                """
                UPDATE queue
                SET dispatched = TRUE
                WHERE id = ?
                """,
                (item[0],),
            )
            return item
        raise RuntimeError("The queue is empty")

    def remove_from_queue(self, item_id: int) -> None:
        """
        Removes an item from the queue
        Args:
            item_id: id of the item to be removed

        Returns: None
        """
        self.cursor.execute(
            """
            DELETE FROM queue
            WHERE id = ?
            """,
            (item_id,),
        )

    def get_queue_size(self):
        """
        Returns the size of the queue
        """
        self.cursor.execute(
            """
            SELECT COUNT(*)
            FROM queue
            WHERE dispatched = FALSE
            """
        )
        return self.cursor.fetchone()[0]

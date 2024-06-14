import orjson as json
from psycopg import connect
from tqdm.auto import tqdm

from config import PSQL_CONN_STRING_SANS_DB


def _copy_into_table(
    copy_context, input_file_path: str, obj_keys: list[str], tqdm_desc: str
) -> None:
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in tqdm(input_file, desc=tqdm_desc, dynamic_ncols=True):
            obj = json.loads(line)
            tup = tuple(obj[key] for key in obj_keys)
            copy_context.write_row(tup)


def _reset_database() -> None:
    conn = connect(PSQL_CONN_STRING_SANS_DB)
    conn.autocommit = True  # drop statements will raise exceptions without autocommit
    c = conn.cursor()
    c.execute("DROP DATABASE IF EXISTS open_library WITH (FORCE)")
    c.execute("CREATE DATABASE open_library OWNER postgres")
    c.close()
    conn.close()


def _load_sql(sql_file_path: str) -> str:
    with open(sql_file_path, "r", encoding="utf-8") as sql_file:
        return sql_file.read()

from kgtk_wrappers import kgtk_query
import pandas as pd
from os import makedirs


def get_clauses(
    source: str, target: str, max_hops: int
) -> list[list[tuple[str, str, str]]]:
    clauses = []
    for i in range(max_hops):
        clauses_max_hops_i = []
        for hop in range(i):
            clause = ""
            clauses.append(clause)
    return clauses


def get_paths(
    input_graph: str,
    graph_cache: str,
    output_dir: str,
    source: str,
    target: str,
    max_hops: int = 3,
) -> pd.DataFrame:
    output_dir_pair = f"{output_dir}/{source}-{target}"
    makedirs(output_dir_pair, exist_ok=True)

    clauses_hops = get_clauses(source, target, max_hops)
    for clauses in clauses_hops:
        for clause in clauses:
            match_cause, where_clause, return_clause = clause
            kgtk_query(
                input_graph=input_graph,
                graph_cache=graph_cache,
                output_path=f"{output_dir_pair}/paths.tsv",
                match_clause=match_cause,
                where_clause=where_clause,
                return_clause=return_clause,
                read_only=True,
            )

    paths = pd.DataFrame()
    return paths

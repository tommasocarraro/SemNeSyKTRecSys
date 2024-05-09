from typing import Sequence, Union
from ..kgtk_wrappers import kgtk_query
from os import makedirs
from joblib import delayed, Parallel
from multiprocessing import cpu_count
import os
import pandas as pd
from .merge_tsv_files import merge_tsv_from_directory
from tqdm.auto import tqdm


def make_where_clause(hops: int) -> str:
    where_clause = ""
    start = 1
    end = start + hops
    for current_node in range(start, end):
        for successor in range(current_node + 1, end):
            where_clause += (
                f"{' and ' if len(where_clause) > 0 else ''}"
                + f"n{current_node} != n{successor}"
            )
    return where_clause


def get_clauses(source: str, target: str, max_hops: int) -> list[tuple[str, str, str]]:
    res = []
    for i in range(1, max_hops + 1):
        node_counter = 2
        rel_counter = 1
        match_clause = f"(n1:{source})"
        where_clause = make_where_clause(i)
        return_clause = "n1 as source, "
        for hop in range(1, i + 1):
            match_clause += f"-[r{rel_counter}]->(n{node_counter}{f':{target}' if hop == i else ''})"
            return_clause += (
                f"r{rel_counter}.label as label{rel_counter}, "
                + f"{f'n{node_counter} as intermediate{node_counter-1}, ' if hop != i else ''}"
            )
            if hop != i:
                node_counter += 1
            rel_counter += 1
        return_clause += f"n{node_counter} as target"
        res.append((match_clause, where_clause, return_clause))
    return res


def run_query_job(
    input_graph: str,
    graph_cache: str,
    output_dir_pair: str,
    clauses: tuple[str, str, str],
    hops: int,
    debug: bool = False,
):
    match_clause, where_clause, return_clause = clauses
    kgtk_query(
        input_graph=input_graph,
        graph_cache=graph_cache,
        output_path=f"{output_dir_pair}/paths_{hops}.tsv",
        match_clause=match_clause,
        where_clause=where_clause,
        return_clause=return_clause,
        read_only=True,
        debug=debug,
    )


def get_paths(
    input_graph: str,
    graph_cache: str,
    output_dir: str,
    source: str,
    target: str,
    max_hops: int = 3,
    debug: bool = False,
    sequential: bool = False,
) -> pd.DataFrame:
    output_dir_pair = os.path.join(output_dir, f"{source}-{target}")
    makedirs(output_dir_pair, exist_ok=True)
    [
        os.remove(os.path.join(output_dir_pair, item))
        for item in os.listdir(output_dir_pair)
    ]

    clauses_hops = get_clauses(source, target, max_hops)

    if sequential:
        for i, clauses in enumerate(clauses_hops):
            run_query_job(
                input_graph=input_graph,
                graph_cache=graph_cache,
                output_dir_pair=output_dir_pair,
                clauses=clauses,
                hops=i,
                debug=debug,
            )
    else:
        Parallel(n_jobs=min(cpu_count(), max_hops), backend="loky")(
            delayed(run_query_job)(
                input_graph, graph_cache, output_dir_pair, clauses, i, debug
            )
            for i, clauses in enumerate(clauses_hops)
        )

    return merge_tsv_from_directory(
        output_dir_pair, os.path.join(output_dir_pair, "paths_all.tsv")
    )


def get_multiple_paths(
    input_graph: str,
    graph_cache: str,
    output_dir: str,
    pairs: Sequence[tuple[str, str]],
    max_hops: int = 3,
    debug: bool = False,
    n_jobs: int = 1,
) -> None:
    with Parallel(n_jobs=n_jobs, backend="loky") as pool:
        pool(
            delayed(get_paths)(
                input_graph,
                graph_cache,
                output_dir,
                source,
                target,
                max_hops,
                debug,
                True,
            )
            for source, target in tqdm(pairs)
        )

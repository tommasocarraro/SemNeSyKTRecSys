import os
from multiprocessing import cpu_count
from typing import Sequence

from joblib import delayed, Parallel
from tqdm.auto import tqdm

from .clauses import get_clauses
from .merge_tsv_files import merge_tsv_from_directory
from ..kgtk_wrappers import kgtk_query


def _run_query_job(
    input_graph: str,
    graph_cache: str,
    output_dir_pair: str,
    clauses: tuple[str, str, str],
    hops: int,
    debug: bool = False,
):
    """
    Wrapper function, required by joblib
    """
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
) -> None:
    """
    Retrieve paths between a source and target node in a graph.

    Args:
        input_graph (str): Path to the input graph file.
        graph_cache (str): Path to the graph cache directory.
        output_dir (str): Path to the output directory.
        source (str): Source node.
        target (str): Target node.
        max_hops (int, optional): Maximum number of hops to consider. Defaults to 3.
        debug (bool, optional): Enable debug mode. Defaults to False.
        sequential (bool, optional): Run queries sequentially. Defaults to False.
    """
    output_dir_pair = os.path.join(output_dir, f"{source}-{target}")
    os.makedirs(output_dir_pair, exist_ok=True)
    [
        os.remove(os.path.join(output_dir_pair, item))
        for item in os.listdir(output_dir_pair)
    ]

    clauses_hops = get_clauses(source, target, max_hops)

    if sequential:
        for i, clauses in enumerate(clauses_hops):
            _run_query_job(
                input_graph=input_graph,
                graph_cache=graph_cache,
                output_dir_pair=output_dir_pair,
                clauses=clauses,
                hops=i,
                debug=debug,
            )
    else:
        Parallel(n_jobs=min(cpu_count(), max_hops), backend="loky")(
            delayed(_run_query_job)(
                input_graph, graph_cache, output_dir_pair, clauses, i, debug
            )
            for i, clauses in enumerate(clauses_hops)
        )

    merge_tsv_from_directory(
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
    gen_len: int = None,
) -> None:
    """
    Given a list of pairs (source, target), generate all paths between pairs.

    Args:
        input_graph (str): The path to the input graph file.
        graph_cache (str): The path to the graph cache file.
        output_dir (str): The directory to store the generated paths.
        pairs (Sequence[tuple[str, str]]): A sequence of node pairs for which paths need to be generated.
        max_hops (int, optional): The maximum number of hops allowed in a path. Defaults to 3.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        n_jobs (int, optional): The number of parallel jobs to run. Defaults to 1.
        gen_len (int, optional): The length of the generator yielding wikidata ID pairs. Defaults to None.
    """
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
            for source, target in tqdm(
                pairs, total=len(pairs) if gen_len is None else gen_len
            )
        )

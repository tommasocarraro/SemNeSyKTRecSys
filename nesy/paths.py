from .kgtk_wrappers import kgtk_query
import pandas as pd
from os import makedirs
from joblib import delayed, Parallel
from multiprocessing import cpu_count
import os
import pandas as pd


# Function to read TSV files from a directory and return a DataFrame with the longest header
def read_longest_header_from_directory(directory: str) -> pd.DataFrame:
    # Initialize variables to store longest header and its corresponding DataFrame
    max_columns = 0
    longest_header_df: pd.DataFrame

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tsv") and not filename.endswith("all.tsv"):
            # Read the file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep="\t")

            # Check if the number of columns in the current file is greater than the maximum
            if len(df.columns) > max_columns:
                max_columns = len(df.columns)
                longest_header_df = df

    return longest_header_df


# Function to merge TSV files from a directory while aligning columns
def merge_tsv_from_directory(directory: str, output_path: str) -> pd.DataFrame:
    # Read the DataFrame with the longest header
    merged_df = read_longest_header_from_directory(directory)

    # Initialize an empty list to store DataFrames to concatenate
    dfs_to_concat = []

    # Iterate through each file in the directory again
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            # Read the file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep="\t")

            # Align columns
            for column in merged_df.columns:
                # If column is missing in the current DataFrame, insert a new column filled with tabs
                if column not in df.columns:
                    df[column] = float("nan") * len(df)
                    # df[column] = ["\t"] * len(df)

            # Reorder columns to match the longest header DataFrame
            df = df.reindex(columns=merged_df.columns)

            # Append current DataFrame to the list
            dfs_to_concat.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs_to_concat, ignore_index=True)
    merged_df.to_csv(output_path, sep="\t", index=False)

    return merged_df


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
) -> pd.DataFrame:
    output_dir_pair = os.path.join(output_dir, f"{source}-{target}")
    makedirs(output_dir_pair, exist_ok=True)
    [
        os.remove(os.path.join(output_dir_pair, item))
        for item in os.listdir(output_dir_pair)
    ]

    clauses_hops = get_clauses(source, target, max_hops)
    Parallel(n_jobs=min(cpu_count(), max_hops), backend="loky")(
        delayed(run_query_job)(
            input_graph, graph_cache, output_dir_pair, clauses, i, debug
        )
        for i, clauses in enumerate(clauses_hops)
    )

    return merge_tsv_from_directory(
        output_dir_pair, os.path.join(output_dir_pair, "paths_all.tsv")
    )

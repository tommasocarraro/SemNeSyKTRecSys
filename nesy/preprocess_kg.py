import os
import shutil
from os import makedirs
from os.path import basename, dirname, join
from nesy.utils import compute_graph_extension
from nesy.kgtk_wrappers import (
    kgtk_add_id,
    kgtk_build_cache,
    kgtk_cat,
    kgtk_filter,
    kgtk_query,
)
from nesy.utils import compute_graph_extension, remove_ext


def preprocess_kg(
    input_graph: str,
    cache_path: str,
    debug: bool,
    index_mode: str = "mode:graph",
    compress_inter_steps: bool = False,
    save_space: bool = True,
    selected_properties: list = None,
):
    """
    Preprocesses the knowledge graph by performing the following steps:
    1. Imports the graph into a temporary cache.
    2. Extracts all properties and computes their inverse.
    3. Generates IDs for the inversed properties.
    4. Concatenates the original and the inversed properties.
    5. Builds the graph cache.
    6. Cleans up temporary files.

    Args:
        input_graph (str): Path to the input graph file.
        cache_path (str): Path to the graph cache file.
        debug (bool): Flag indicating whether to enable debug mode.
        index_mode (str, optional): Index mode for building the graph cache. Defaults to "mode:graph".
        compress_inter_steps (bool, optional): Flag indicating whether to compress temporary files during intermediate. Defaults to False.
        save_space (bool, optional): Flag indicating whether to delete the temporary files as soon as they are not needed. Defaults to True
        selected_properties (list, optional): List of properties that has to be inserted in the graph cache. Defaults to None, meaning all properties are inserted.

    Returns:
        The path to the preprocessed graph.
    """
    graph_name = remove_ext(basename(input_graph))
    base_temp_dir = join(dirname(input_graph), "tmp")
    makedirs(base_temp_dir, exist_ok=True)

    # optionally use filter to remove unwanted properties
    filtered_graph = None
    if selected_properties is not None:
        print("Filtering out unwanted properties")
        filtered_graph = join(
            base_temp_dir,
            graph_name + "_filtered" + compute_graph_extension(compress_inter_steps),
        )
        pattern = " : %s : " % ("|".join(selected_properties))
        kgtk_filter(
            input_graph=input_graph,
            output_path=filtered_graph,
            word_separator="|",
            pattern_separator=":",
            invert=False,
            pattern=pattern,
        )
        # modify the input graph to be the new filtered graph
        input_graph = filtered_graph

    print("Importing graph into temporary cache")
    temp_graph_cache = join(base_temp_dir, "temp_cache.sqlite3.db")
    kgtk_build_cache(input_graph=input_graph, graph_cache=temp_graph_cache, debug=debug)

    print("Importing graph into temporary cache")
    temp_graph_cache = join(base_temp_dir, "temp_cache.sqlite3.db")
    kgtk_build_cache(
        input_graph=input_graph if filtered_graph is None else filtered_graph,
        graph_cache=temp_graph_cache,
        debug=debug,
    )

    print("Extracting all properties and computing their inverse")
    output_inverse_graph = join(
        base_temp_dir,
        graph_name + "_inverse" + compute_graph_extension(compress_inter_steps),
    )
    kgtk_query(
        input_graph=input_graph if filtered_graph is None else filtered_graph,
        match_clause="(n1)-[r1]->(n2)",
        where_clause='r1 != "P155" and r1 != "P156" and r1 != "P1365" and r1 != "P1366"',
        return_clause="n2 as node1, CONCAT(r1.label, '_') as label, n1 as node2",
        output_path=output_inverse_graph,
        debug=debug,
        graph_cache=temp_graph_cache,
    )
    if save_space:
        os.remove(temp_graph_cache)
        if filtered_graph is not None:
            os.remove(filtered_graph)

    print("Generating IDs for the inversed properties")
    output_inverse_ids_graph = join(
        base_temp_dir,
        graph_name
        + "_inverse_with_ids"
        + compute_graph_extension(compress_inter_steps),
    )
    kgtk_add_id(input_graph=output_inverse_graph, output_path=output_inverse_ids_graph)
    if save_space:
        os.remove(output_inverse_graph)

    print("Concatenating the original and the inversed properties")
    concatenated_graph = join(dirname(input_graph), graph_name + "_preprocessed.tsv.gz")
    kgtk_cat(
        input_graphs=[input_graph, output_inverse_ids_graph],
        output_path=concatenated_graph,
    )
    if save_space:
        os.remove(output_inverse_ids_graph)

    print("Building the graph cache")
    kgtk_build_cache(
        input_graph=concatenated_graph,
        graph_cache=cache_path,
        index_mode=index_mode,
        debug=debug,
    )

    # removing temp directory
    shutil.rmtree(base_temp_dir)

    print(f"Final knowledge graph saved at {concatenated_graph}")

    return concatenated_graph


# # Example usage
# kg = "../data/wikidata/claims.wikibase-item.tsv.gz"
# cache = "../data/wikidata/graph-cache.sqlite3.db"
# preprocess_kg(input_graph=kg, cache_path=cache, compress_inter_steps=True, debug=True)

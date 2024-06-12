def match_step(
    target: str, hop: int, max_hops: int, rel_counter: int, node_counter: int
) -> str:
    """
    Constructs a MATCH step for a KGTK query.

    Args:
        target (str): The target node label or property.
        hop (int): The current hop number.
        max_hops (int): The maximum number of hops allowed.
        rel_counter (int): The counter for relationship variables.
        node_counter (int): The counter for node variables.

    Returns:
        str: The MATCH step for the KGTK query.
    """
    return (
        f"-[r{rel_counter}]->(n{node_counter}{f':{target}' if hop == max_hops else ''})"
    )


def return_step(rel_counter: int, node_counter: int, hop: int, max_hops: int) -> str:
    """
    Constructs a RETURN step for a KGTK query.

    Args:
        rel_counter (int): The counter for the relationship.
        node_counter (int): The counter for the node.
        hop (int): The current hop in the path.
        max_hops (int): The maximum number of hops in the path.

    Returns:
        str: The RETURN step for the KGTK query.
    """
    return (
        f"r{rel_counter}.label as label{rel_counter}, "
        + f"{f'n{node_counter} as intermediate{node_counter-1}, ' if hop != max_hops else ''}"
    )


def make_clauses_max_hops_i(
    source: str, target: str, max_hops: int
) -> tuple[str, str, str]:
    """
    Generate clauses for finding paths between a source and target node
    with a maximum number of hops.

    Args:
        source (str): The label of the source node.
        target (str): The label of the target node.
        max_hops (int): The maximum number of hops allowed in the path.

    Returns:
        tuple[str, str, str]: A tuple containing the match clause, where clause,
        and return clause for the path query.
    """
    node_counter = 2
    rel_counter = 1
    match_clause = f"(n1:{source})"
    where_clause = ""
    return_clause = "n1 as source, "
    for hop in range(1, max_hops + 1):
        # compute the where step
        for successor in range(hop + 1, max_hops + 1):
            where_clause += (
                f"{' and ' if len(where_clause) > 0 else ''}"
                + f"n{hop} != n{successor}"
            )

        match_clause += match_step(target, hop, max_hops, rel_counter, node_counter)

        return_clause += return_step(rel_counter, node_counter, hop, max_hops)
        if hop != max_hops:
            node_counter += 1
        rel_counter += 1
    return_clause += f"n{node_counter} as target"
    return match_clause, where_clause, return_clause


def get_clauses(source: str, target: str, max_hops: int) -> list[tuple[str, str, str]]:
    """
    Generate a list of clauses for a given source, target, and maximum number of hops.

    Args:
        source (str): The source node.
        target (str): The target node.
        max_hops (int): The maximum number of hops.

    Returns:
        list[tuple[str, str, str]]: A list of tuples containing the match, where, and return clauses.
    """
    return [make_clauses_max_hops_i(source, target, i) for i in range(1, max_hops + 1)]

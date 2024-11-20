def make_query(
    max_hops: int, shortest_path: bool, source_domain: str, target_domain: str
) -> str:
    """
    Creates the query for Neo4j based on the given number of hops.

    :param max_hops: max number of hops allowed for the path
    :param shortest_path: whether to configure the query to find just the shortest path or not
    :param source_domain: source domain name
    :param target_domain: target domain name
    :return: the query to be executed
    """
    query_head = "(n1:entity {wikidata_id: $first_item})"
    query_tail = "(n2:entity {wikidata_id: $second_item})"
    if not shortest_path:
        query = ""
        for i in range(max_hops):
            query += "MATCH path=%s" % (query_head,)
            for j in range(i):
                query += "-[*1..1]-(mid%d:entity)" % (j + 1,)
            query += "-[*1..1]-"
            query += "%s RETURN path, length(path) AS path_length" % (query_tail,)
            if i != max_hops - 1:
                query += " UNION "
    else:
        # this approach only works for shortest paths as it computes only a single path per pair
        # first it checks if the special path already exists, if it doesn't then it is computed
        query = f"""
            MATCH (n1:entity {{wikidata_id: $first_item}})
            MATCH (n2:entity {{wikidata_id: $second_item}})

            WITH n1, n2
            OPTIONAL MATCH path = (n1)-[r:precomputed]-(n2)
            WITH n1, n2, path

            WHERE path IS NULL
            MATCH p=shortestPath((n1)-[r:relation*1..{max_hops}]-(n2))
            WITH n1, n2, length(p) AS pathLength, [n IN nodes(p) | n] AS nodeList, [r IN relationships(p) | r] AS relList
            WITH n1, n2, pathLength, reduce(s = "", i IN range(0, size(relList)-1) |
                s + 
                case 
                    when i = 0 
                    then "(" + nodeList[0].label + ": " + nodeList[0].wikidata_id + ")"
                    else "" 
                end +
                case 
                    when startNode(relList[i]) = nodeList[i] 
                    then "-"
                    else "<-"
                end +
                "[" + relList[i].label + ": " + relList[i].wikidata_id + "]" +
                case 
                    when endNode(relList[i]) = nodeList[i+1] 
                    then "->"
                    else "-"
                end +
                "(" + nodeList[i+1].label + ": " + nodeList[i+1].wikidata_id + ")"
            ) AS pathString
            CREATE (n1)-[r1:precomputed {{path_length: pathLength, path_string: pathString, source_domain: {"'" + source_domain + "'"}, target_domain: {"'" + target_domain + "'"}}}]->(n2)
            """
    return query

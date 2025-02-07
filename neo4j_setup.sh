#!/bin/bash
set -e

neo4j-admin dbms set-initial-password your_password

neo4j start

echo "Importing database..."
neo4j-admin database import full wikidata --nodes=/import/nodes.csv --relationships=/import/relationships.csv --overwrite-destination --verbose

echo "Waiting for Neo4j to be available..."
until cypher-shell -u neo4j -p your_password "RETURN 1;" >/dev/null 2>&1; do
    sleep 2
done

cypher-shell -u neo4j -p your_password "CREATE DATABASE wikidata IF NOT EXISTS;"

echo "Creating indexes..."
cypher-shell -u neo4j -p your_password "CREATE TEXT INDEX node_wikidata_id_text_index IF NOT EXISTS FOR (n:entity) ON (n.wikidata_id);"
until cypher-shell -u neo4j -p your_password "SHOW INDEXES;" | grep -q "node_wikidata_id_text_index.*ONLINE"; do
    sleep 2 # loop until the index has been created
done

echo "Neo4j setup complete."
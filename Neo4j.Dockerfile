FROM neo4j:5.26.1-enterprise

# install micro in case conf modifications are needed afterwards
RUN apt-get update && apt-get install -y micro

# run the import command and create a file when finished to synchronize with the python app
CMD neo4j-admin database import full wikidata --nodes=/import/nodes.csv --relationships=/import/relationships.csv --report-file=/dev/null --overwrite-destination && touch /import/import_done
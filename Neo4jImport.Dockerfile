FROM neo4j:5.26.1-enterprise

COPY ./data/wikidata/nodes.csv /import/nodes.csv
COPY ./data/wikidata/relationships.csv /import/relationships.csv

COPY neo4j_setup.sh /neo4j_setup.sh
RUN chmod +x /neo4j_setup.sh

CMD ["/neo4j_setup.sh"]
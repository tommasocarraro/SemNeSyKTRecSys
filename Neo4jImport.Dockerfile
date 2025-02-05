FROM neo4j:5.26.1-enterprise

COPY neo4j_setup.sh /neo4j_setup.sh
RUN chmod +x /neo4j_setup.sh

CMD ["/neo4j_setup.sh"]
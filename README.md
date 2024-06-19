# SemNeSyKTRecSys
Semantic Neuro-Symbolic Knowledge Transfer for Recommender Systems

## TODO list:
- owsky:
  - improve type hints
  - write comments
  - implement better api results filtering and selection strategy
- ciomi:
  - better check wikidata categories to create the small dumps used to filter out unwanted matches (especially music that has only album for the moment)
  - code refactoring to put code in the right module
  - finish the first scraping loop with Amazon and then use API to complete data
  - implement driver to launch cypher queries from python on neo4j
  - understand how to rank paths and the desired path length
  - import of the new wikidata file (check filtered relationships carefully)
  - import labels and put labels in paths
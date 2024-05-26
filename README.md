# SemNeSyKTRecSys
Semantic Neuro-Symbolic Knowledge Transfer for Recommender Systems

## TODO list:
- Implement better HTTP 429 error handling, i.e., stop querying the APIs, process data received up to that point, sleep for a certain amount of time and retry
- Implement batching for queries
- Figure out why google kg search for books is duplicating warning logs
- better check wikidata categories to create the small dumps used to filter out unwanted matches
- code refactoring to put code in the right module
- is it better to move to Neo4j?
- try to look for date before removing parentheses
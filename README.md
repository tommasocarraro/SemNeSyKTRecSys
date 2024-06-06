# SemNeSyKTRecSys
Semantic Neuro-Symbolic Knowledge Transfer for Recommender Systems

## TODO list:
- better check wikidata categories to create the small dumps used to filter out unwanted matches
- code refactoring to put code in the right module
- google scraping should be improved by checking if the page is an Amazon page and include the ASIN in the URL
- improve type hints
- write comments
- improve upon local search on open library:
  - check against person or year, too if available
  - tie in with the rest of api querying program
- Books:
  - First query on KG graph and try to find a plausible match
  - Then if information is still missing try to complete with open library
- Overhaul metadata preprocessing:
  - Include original and cleaned title
  - Save person as array
- Metadata after matching against APIs:
  - show where each piece of information came from
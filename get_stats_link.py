import pandas as pd
import json


# Function to compute statistics for a given dataset
def compute_wikidata_statistics(ratings_file, metadata_file):
    # Load the rating file (CSV)
    ratings_df = pd.read_csv(ratings_file)

    # Load the metadata file (JSON)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Number of unique items in the ratings file
    num_items = ratings_df['itemId'].nunique()

    # Initialize counters
    linked_count = 0
    count_title_person_year = 0
    count_title_person = 0
    count_title_year = 0
    count_title = 0

    # Iterate over the metadata dictionary
    for item_id, data in metadata.items():
        if isinstance(data, dict) and 'wiki_id' in data:  # Valid Wikidata entry
            linked_count += 1
            matching_attributes = data.get('matching_attributes', '')
            if 'title' in matching_attributes and 'person' in matching_attributes and 'year' in matching_attributes:
                count_title_person_year += 1
            elif 'title' in matching_attributes and 'person' in matching_attributes:
                count_title_person += 1
            elif 'title' in matching_attributes and 'year' in matching_attributes:
                count_title_year += 1
            elif 'title' in matching_attributes:
                count_title += 1

    # Calculate percentages
    linked_percentage = (linked_count / num_items) * 100
    title_person_year_percentage = (count_title_person_year / linked_count) * 100 if linked_count > 0 else 0
    title_person_percentage = (count_title_person / linked_count) * 100 if linked_count > 0 else 0
    title_year_percentage = (count_title_year / linked_count) * 100 if linked_count > 0 else 0
    title_percentage = (count_title / linked_count) * 100 if linked_count > 0 else 0

    return num_items, linked_count, linked_percentage, count_title_person_year, title_person_year_percentage, count_title_person, title_person_percentage, count_title_year, title_year_percentage, count_title, title_percentage


# Datasets with their corresponding rating and metadata files
datasets = {
    'Books': ('./data/processed/legacy/reviews_Books_5.csv', './data/processed/mappings/mapping-books.json'),
    'Music': ('./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv', './data/processed/mappings/mapping-music.json'),
    'Movies': ('./data/processed/legacy/reviews_Movies_and_TV_5.csv', './data/processed/mappings/mapping-movies.json')
}

# Compute statistics for each dataset
stats = {name: compute_wikidata_statistics(ratings_file, metadata_file)
         for name, (ratings_file, metadata_file) in datasets.items()}

# Generate LaTeX table with formatted percentages
latex_table = r"""
\begin{table}[htbp]
    \centering
    \begin{tabular}{lccc}
    \hline
    Statistic & Books & Music & Movies \\
    \hline
    Number of Items Linked to Wikidata & %d (%.2f) & %d (%.2f) & %d (%.2f) \\
    Number of Links with Title, Person, and Year & %d (%.2f) & %d (%.2f) & %d (%.2f) \\
    Number of Links with Title and Person & %d (%.2f) & %d (%.2f) & %d (%.2f) \\
    Number of Links with Title and Year & %d (%.2f) & %d (%.2f) & %d (%.2f) \\
    Number of Links with Title Only & %d (%.2f) & %d (%.2f) & %d (%.2f) \\
    \hline
    \end{tabular}
    \caption{Wikidata Linking Statistics for Books, Music, and Movies Datasets}
    \label{tab:wikidata_stats}
\end{table}
""" % (
    stats['Books'][1], stats['Books'][2],
    stats['Music'][1], stats['Music'][2],
    stats['Movies'][1], stats['Movies'][2],
    stats['Books'][3], stats['Books'][4],
    stats['Music'][3], stats['Music'][4],
    stats['Movies'][3], stats['Movies'][4],
    stats['Books'][5], stats['Books'][6],
    stats['Music'][5], stats['Music'][6],
    stats['Movies'][5], stats['Movies'][6],
    stats['Books'][7], stats['Books'][8],
    stats['Music'][7], stats['Music'][8],
    stats['Movies'][7], stats['Movies'][8],
    stats['Books'][9], stats['Books'][10],
    stats['Music'][9], stats['Music'][10],
    stats['Movies'][9], stats['Movies'][10]
)

# Output the LaTeX table
print(latex_table)
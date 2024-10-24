import pandas as pd
import json


# Function to compute statistics for a given dataset
def compute_metadata_statistics(ratings_file, metadata_file):
    # Load the rating file (CSV)
    ratings_df = pd.read_csv(ratings_file)

    # Load the metadata file (JSON)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Number of unique items in the ratings file
    num_items = ratings_df['itemId'].nunique()

    # Initialize counters for available metadata
    title_count = 0
    person_count = 0
    year_count = 0

    # Iterate over the metadata dictionary
    for item_id, data in metadata.items():
        if data != "404-error":  # Valid metadata entry
            if data.get('title'):  # Check if title is available
                title_count += 1
            if data.get('person'):  # Check if person is available
                person_count += 1
            if data.get('year'):  # Check if year is available
                year_count += 1

    # Calculate percentages
    title_percentage = (title_count / num_items) * 100
    person_percentage = (person_count / num_items) * 100
    year_percentage = (year_count / num_items) * 100

    return num_items, title_count, title_percentage, person_count, person_percentage, year_count, year_percentage


# Datasets with their corresponding rating and metadata files
datasets = {
    'Books': ('./data/processed/legacy/reviews_Books_5.csv', './data/processed/complete-metadata/complete-books.json'),
    'Music': ('./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv', './data/processed/complete-metadata/complete-music.json'),
    'Movies': ('./data/processed/legacy/reviews_Movies_and_TV_5.csv', './data/processed/complete-metadata/complete-movies.json')
}


# Compute statistics for each dataset
stats = {name: compute_metadata_statistics(ratings_file, metadata_file)
         for name, (ratings_file, metadata_file) in datasets.items()}

# Generate LaTeX table with classic formatting
latex_table = r"""
\begin{table}[htbp]
    \centering
    \begin{tabular}{lccc}
    \hline
    Statistic & Books & Music & Movies \\
    \hline
    Number of Items & %d & %d & %d \\
    Number of Titles Available & %d & %d & %d \\
    Percentage of Titles Available & %.2f & %.2f & %.2f \\
    Number of Persons Available & %d & %d & %d \\
    Percentage of Persons Available & %.2f & %.2f & %.2f \\
    Number of Years Available & %d & %d & %d \\
    Percentage of Years Available & %.2f & %.2f & %.2f \\
    \hline
    \end{tabular}
    \caption{Metadata Availability Statistics for Books, Music, and Movies Datasets}
    \label{tab:metadata_stats}
\end{table}
""" % (
    stats['Books'][0], stats['Music'][0], stats['Movies'][0],
    stats['Books'][1], stats['Music'][1], stats['Movies'][1],
    stats['Books'][2], stats['Music'][2], stats['Movies'][2],
    stats['Books'][3], stats['Music'][3], stats['Movies'][3],
    stats['Books'][4], stats['Music'][4], stats['Movies'][4],
    stats['Books'][5], stats['Music'][5], stats['Movies'][5],
    stats['Books'][6], stats['Music'][6], stats['Movies'][6]
)

# Output the LaTeX table
print(latex_table)
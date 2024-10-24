import pandas as pd
import json


def calculate_stats(rating_file):
    # Load the rating file (CSV)
    ratings_df = pd.read_csv(rating_file)

    # Load the metadata file (JSON)
    with open('./data/processed/legacy/metadata.json', 'r') as f:
        metadata = json.load(f)

    # Number of users
    num_users = ratings_df['userId'].nunique()

    # Number of items
    num_items = ratings_df['itemId'].nunique()

    # Number of ratings
    num_ratings = len(ratings_df)

    # Density of ratings (number of ratings / total possible ratings)
    density = (num_ratings / (num_users * num_items)) * 100

    # Number of available titles (from metadata)
    # Filter the unique itemIds in the ratings that have a corresponding title in the metadata
    available_item_ids = ratings_df['itemId'].unique()
    available_titles = [item_id for item_id in available_item_ids if metadata[item_id] != "no-title"]
    num_available_titles = len(available_titles)

    # Percentage of available titles
    percentage_available_titles = (num_available_titles / num_items) * 100

    # Return the calculated statistics
    return {
        'num_users': num_users,
        'num_items': num_items,
        'num_ratings': num_ratings,
        'density': density,
        'num_available_titles': num_available_titles,
        'percentage_available_titles': percentage_available_titles
    }


# Datasets
datasets = {
    'Books': './data/processed/legacy/reviews_Books_5.csv',
    'Music': './data/processed/legacy/reviews_CDs_and_Vinyl_5.csv',
    'Movies': './data/processed/legacy/reviews_Movies_and_TV_5.csv'
}

# Calculate statistics for each dataset
stats = {name: calculate_stats(ratings_file) for name, ratings_file in datasets.items()}

# Generate LaTeX table
latex_table = r"""
\begin{table}[htbp]
    \centering
    \begin{tabular}{lccc}
    \hline
    Statistic & Books & Music & Movies \\
    \hline
    Number of Users & %d & %d & %d \\
    Number of Items & %d & %d & %d \\
    Number of Ratings & %d & %d & %d \\
    Density of Ratings & %.4f & %.4f & %.4f \\
    Number of Available Titles & %d & %d & %d \\
    Percentage of Available Titles & %.2f & %.2f & %.2f \\
    \hline
    \end{tabular}
    \caption{Statistics of the Books, Music, and Movies Datasets}
    \label{tab:dataset_stats}
\end{table}
""" % (
    stats['Books']['num_users'], stats['Music']['num_users'], stats['Movies']['num_users'],
    stats['Books']['num_items'], stats['Music']['num_items'], stats['Movies']['num_items'],
    stats['Books']['num_ratings'], stats['Music']['num_ratings'], stats['Movies']['num_ratings'],
    stats['Books']['density'], stats['Music']['density'], stats['Movies']['density'],
    stats['Books']['num_available_titles'], stats['Music']['num_available_titles'], stats['Movies']['num_available_titles'],
    stats['Books']['percentage_available_titles'], stats['Music']['percentage_available_titles'], stats['Movies']['percentage_available_titles']
)

# Output the LaTeX table
print(latex_table)


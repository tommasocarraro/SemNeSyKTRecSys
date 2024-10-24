import pandas as pd
import json


# Function to filter items linked to the knowledge graph and based on rating counts
def filter_linked_items_by_ratings(df, path_data, min_ratings=None, max_ratings=None, dataset_type='first'):
    item_counts = df['itemId'].value_counts()
    if dataset_type == 'first':
        linked_items = [item for item in item_counts.index if item in path_data]
    else:
        linked_items = [item for item in item_counts.index if any(item in v for v in path_data.values())]

    if min_ratings is not None:
        return [item for item in linked_items if item_counts[item] >= min_ratings]
    if max_ratings is not None:
        return [item for item in linked_items if item_counts[item] <= max_ratings]


# Function to count common users between two datasets
def count_common_users(df1, df2):
    users1 = set(df1['userId'].unique())
    users2 = set(df2['userId'].unique())
    return len(users1.intersection(users2))


# Function to count paths between high and low rated items
def count_paths_between_items(paths, high_rating_items, low_rating_items):
    count = 0
    for high_item in high_rating_items:
        if high_item in paths:
            for low_item in low_rating_items:
                if low_item in paths[high_item] and isinstance(paths[high_item][low_item], list):
                    count += 1
    return count


# LaTeX table header
latex_table = r"""
\begin{table}[h]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
Dataset Pair & Items (>= 300 Ratings) & Items (<= 5 Ratings) & Common Users & Total Pairs & Paths Found & \% Paths Found \\
\hline
"""

# Dataset pairs and their corresponding path files
dataset_pairs = [
    ("books", "movies", "./data/processed/paths/paths/books(pop:300)-movies(cs:5).json"),
    ("books", "music", "./data/processed/paths/books(pop:300)-music(cs:5).json"),
    ("movies", "books", "./data/processed/paths/paths/movies(pop:300)-books(cs:5).json"),
    ("movies", "music", "./data/processed/paths/movies(pop:300)-music(cs:5).json"),
    ("music", "books", "./data/processed/paths/paths/music(pop:300)-books(cs:5).json"),
    ("music", "movies", "./data/processed/paths/paths/music(pop:300)-movies(cs:5).json")
]

# Ratings data
ratings_files = {
    "books": pd.read_csv("./data/processed/legacy/reviews_Books_5.csv"),
    "music": pd.read_csv("./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv"),
    "movies": pd.read_csv("./data/processed/legacy/reviews_Movies_and_TV_5.csv")
}

# Iterate over dataset pairs and compute statistics
for first_dataset, second_dataset, path_file in dataset_pairs:
    # Load the path file (assumed to be in JSON format)
    with open(path_file, 'r') as f:
        paths = json.load(f)

    # Get items with >= 300 ratings linked in the first dataset
    items_with_high_ratings = filter_linked_items_by_ratings(ratings_files[first_dataset],
                                                             paths,
                                                             min_ratings=300, dataset_type='first')

    # Get items with <= 5 ratings linked in the second dataset
    items_with_low_ratings = filter_linked_items_by_ratings(ratings_files[second_dataset],
                                                            paths,
                                                            max_ratings=5, dataset_type='second')

    # Compute number of common users between the two datasets
    common_users = count_common_users(ratings_files[first_dataset], ratings_files[second_dataset])

    # Compute the product of the number of items
    total_pairs = len(items_with_high_ratings) * len(items_with_low_ratings)

    # Count the number of paths found
    paths_found = count_paths_between_items(paths, items_with_high_ratings, items_with_low_ratings)

    # Compute percentage of paths found
    percentage_paths_found = (paths_found / total_pairs * 100) if total_pairs > 0 else 0

    # Add this row to the LaTeX table
    latex_table += f"{first_dataset.capitalize()} --> {second_dataset.capitalize()} & {len(items_with_high_ratings)} & {len(items_with_low_ratings)} & {common_users} & {total_pairs} & {paths_found} & {percentage_paths_found:.2f}\\% \\\\\n"

# LaTeX table footer
latex_table += r"""
\hline
\end{tabular}
}
\caption{Path Statistics for Dataset Pairs}
\end{table}
"""

# Output the LaTeX table
print(latex_table)

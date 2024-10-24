import json
from collections import defaultdict
import pandas as pd


# Function to load a JSON path file from a given file path
def load_path_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Function to count path lengths for a single path file
def count_paths_by_length(path_file):
    path_length_counts = defaultdict(int)

    for source_item, target_items in path_file.items():
        for target_item, paths in target_items.items():
            if isinstance(paths, list):  # Check if paths exist
                for path_info in paths:
                    path_length = path_info.get('path_length', 0)
                    if 1 <= path_length <= 10:  # Valid path length range
                        path_length_counts[path_length] += 1

    return path_length_counts


# Function to compute path length statistics for all dataset pairs
# Function to compute path length statistics for all dataset pairs and combine into a single table
# Function to compute path length statistics for all dataset pairs and combine into a single table
def compute_combined_path_statistics(dataset_pairs):
    all_data = defaultdict(list)
    max_rows = 0  # Keep track of the max number of rows for padding

    for dataset_pair, path_file_path in dataset_pairs.items():
        # Load the path file from the file path
        path_file = load_path_file(path_file_path)

        # Count paths by length for the current pair's path file
        path_length_counts = count_paths_by_length(path_file)

        # Compute total number of paths
        total_paths = sum(path_length_counts.values())

        # Collect statistics for each path length (and total)
        dataset_stats = []
        dataset_stats.append(f"{total_paths}")  # Total paths

        for length in range(1, 11):
            count = path_length_counts.get(length, 0)
            percentage = (count / total_paths) * 100 if total_paths > 0 else 0
            dataset_stats.append(f"{count} ({percentage:.2f}%)")

        # Update the max number of rows if this dataset has more rows
        max_rows = max(max_rows, len(dataset_stats))

        # Append statistics to the corresponding dataset pair column
        all_data[dataset_pair] = dataset_stats

    # Ensure all columns have the same number of rows (padding with empty values if necessary)
    for dataset_pair in all_data:
        while len(all_data[dataset_pair]) < max_rows:
            all_data[dataset_pair].append('')  # Pad with empty strings

    # Prepare row labels for the table (total and path lengths)
    row_labels = ['Total Paths Found'] + [f"Path Length {i}" for i in range(1, 11)]
    all_data['Statistic'] = row_labels

    # Convert the collected data into a pandas DataFrame for LaTeX export
    return pd.DataFrame(all_data)


# Example of dataset pairs with file paths
dataset_pairs = {
    "Books --> Movies": "./data/processed/paths/paths/books(pop:300)-movies(cs:5).json",
    "Books --> Music": "./data/processed/paths/books(pop:300)-music(cs:5).json",
    "Movies --> Books": "./data/processed/paths/paths/movies(pop:300)-books(cs:5).json",
    "Movies --> Music": "./data/processed/paths/movies(pop:300)-music(cs:5).json",
    "Music --> Books": "./data/processed/paths/paths/music(pop:300)-books(cs:5).json",
    "Music --> Movies": "./data/processed/paths/paths/music(pop:300)-movies(cs:5).json",
}

# Compute the combined path statistics
combined_df = compute_combined_path_statistics(dataset_pairs)

# Generate a single LaTeX table with all dataset pairs
latex_table = combined_df.to_latex(index=False, escape=False)
print(latex_table)

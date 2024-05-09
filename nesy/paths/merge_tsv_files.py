import os
import pandas as pd


# Function to read TSV files from a directory and return a DataFrame with the longest header
def read_longest_header_from_directory(directory: str) -> pd.DataFrame:
    # Initialize variables to store longest header and its corresponding DataFrame
    max_columns = 0
    longest_header_df: pd.DataFrame

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tsv") and not filename.endswith("all.tsv"):
            # Read the file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep="\t")

            # Check if the number of columns in the current file is greater than the maximum
            if len(df.columns) > max_columns:
                max_columns = len(df.columns)
                longest_header_df = df

    return longest_header_df


# Function to merge TSV files from a directory while aligning columns
def merge_tsv_from_directory(directory: str, output_path: str) -> pd.DataFrame:
    # Read the DataFrame with the longest header
    merged_df = read_longest_header_from_directory(directory)

    # Initialize an empty list to store DataFrames to concatenate
    dfs_to_concat = []

    # Iterate through each file in the directory again
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            # Read the file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep="\t")

            # Align columns
            for column in merged_df.columns:
                # If column is missing in the current DataFrame, insert a new column filled with tabs
                if column not in df.columns:
                    df[column] = float("nan") * len(df)
                    # df[column] = ["\t"] * len(df)

            # Reorder columns to match the longest header DataFrame
            df = df.reindex(columns=merged_df.columns)

            # Append current DataFrame to the list
            dfs_to_concat.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs_to_concat, ignore_index=True)
    merged_df.to_csv(output_path, sep="\t", index=False)

    return merged_df

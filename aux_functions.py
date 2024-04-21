import pandas as pd

def print_contents(df, n, column_name):
    """
    Print the first n entries of the 'contents' column from the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    n (int): The number of entries to print from the 'contents' column.
    """
    if column_name not in df.columns:
        print(f"Error: '{column_name}' is not a valid column name.")
        return
    
    for i in range(n):
        if i < len(df[column_name]):
            print(f"Content {i+1}: {df[column_name][i]}")
        else:
            print(f"Content {i+1}: Index out of range.")
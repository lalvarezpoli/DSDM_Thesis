import pandas as pd
import re

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

def extract_position_and_clean_content(df):
    """
    Extracts the job position of the speaker from the 'contents' column of the DataFrame,
    adds this information to a new column 'position_speaker', and cleans the 'contents' by removing the extracted job position.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with the job position extracted and contents cleaned.
    """
    # Define a regex pattern to find common job positions
    pattern = r"Member of the Executive Board|Vice-President|President"

    # Function to extract job position and clean content
    def extract_position(content):
        match = re.search(pattern, content, re.IGNORECASE)  # Using re.IGNORECASE to handle case insensitivity
        if match:
            position = match.group(0)  # Capture the matched job position
            cleaned_content = re.sub(re.escape(position), "", content, 1).strip()  # Remove the position from content
            return position, cleaned_content
        return None, content  # If no match, return None for position and original content

    # Apply the function to the 'contents' column
    df['position_speaker'], df['contents'] = zip(*df['contents'].apply(extract_position))

    return df


def extract_interviewer(df, column_name='title'):
    """
    Extracts information about the interviewer based on keywords 'with', 'on', 'for' from the specified column
    in the DataFrame, and adds this information to a new column 'interviewer'.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (str): The name of the column from which to extract the interviewer information.

    Returns:
        pd.DataFrame: The DataFrame with the interviewer information added.
    """
    # Define a combined regex pattern that gives priority to 'with', then 'on', then 'for'
    # Added non-capturing groups (?:) to ensure only one group is captured at a time
    pattern = r"(?:with|for|to|on)\s+([^,.]+)"

    # Function to extract interviewer information based on the combined pattern
    def extract_info(text):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Returns the first non-empty capturing group found
            return match.group(1).strip()
        return text # None to remove the exceptions

    # Apply the extraction function to the specified column and create 'interviewer' column
    df['interviewer'] = df[column_name].apply(extract_info)

    return df

def extract_subtitle(df, column_name='subtitle'):
    # Define a combined regex pattern that gives priority to 'with', then 'on', then 'for'
    # Added non-capturing groups (?:) to ensure only one group is captured at a time
    pattern = r"(?:conducted|published)\s+([^.]+)"

    # Function to extract interviewer information based on the combined pattern
    def extract_info(text):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Returns the first non-empty capturing group found
            return match.group(0).strip()
        return text # None to remove the exceptions

    # Apply the extraction function to the specified column and create 'interviewer' column
    df['subtitle'] = df[column_name].apply(extract_info)

    return df

def refine_contents(df):
    # Function to extract the content after the subtitle
    def extract_post_subtitle(contents, subtitle):
        # Escape regex special characters in the subtitle
        pattern = re.escape(subtitle)
        # Find the match for the subtitle in the contents
        match = re.search(pattern, contents, flags=re.IGNORECASE)
        if match:
            # Return everything after the subtitle
            return contents[match.end():].strip()
        return contents  # Return the original contents if no match is found

    # Apply the function to each row using the 'subtitle' and 'contents' columns
    df['contents'] = df.apply(lambda row: extract_post_subtitle(row['contents'], row['subtitle']), axis=1)

    return df

def keep_text_after_date(df, column_name='contents'):

    date_pattern = r"(\d{1,2} [A-Z][a-z]+ \d{4})"  # Pattern to match dates like '24 January 2020'

    def adjust_text(text):
        # Search for the first period to determine the first sentence
        first_period_idx = text.find('.')
        first_sentence = text[:first_period_idx] if first_period_idx != -1 else text
        
        # Search for a date in the first sentence
        match = re.search(date_pattern, first_sentence)
        if match:
            # If a date is found, find its position
            date_end_idx = match.end()
            # Check if this date is within the first sentence
            if date_end_idx <= first_period_idx or first_period_idx == -1:
                # Keep text after the date
                return text[date_end_idx:].strip()
        
        return text  # Return original text if no date in the first sentence or no period found

    # Apply the function to each entry in the specified column
    df[column_name] = df[column_name].apply(adjust_text)
    return df

def remove_initial_dot(df, column_name='contents'):

    def adjust_text(text):
        # Check the first three characters for a dot and remove the first dot found
        if text[:3].find('.') != -1:
            first_dot_idx = text[:3].find('.')
            # Remove the first dot found within the first three characters
            return text[:first_dot_idx] + text[first_dot_idx + 1:]
        return text
    
    # Apply the function to each entry in the specified column
    df[column_name] = df[column_name].apply(adjust_text)
    return df


import re
import ast
import pickle
import Levenshtein
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ecbdata import ecbdata
from fuzzywuzzy import fuzz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

from openai import OpenAI
from jinja2 import Template

tqdm.pandas()

#ECB Color
color = (17/255, 49/255, 147/255)

# Functions:
# Function to count the frequency of words from the list in a sentence
def count_frequency(answer, words_to_match):
    if pd.isna(answer):
        return 0
    sentence_lower = answer.lower()
    return sum(sentence_lower.count(word) for word in words_to_match)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_end_answer(text):
    text = text.replace(r'[end_answer]', '')
    return text

def remove_special_characters_keep_point(text):
    # Define the regex pattern to match special characters except "."
    pattern = r"[^\w\s\.\'\!\?]"
    # Replace special characters with an empty string
    text = re.sub(pattern, '', text)
    # Replace consecutive dots with just one dot
    text = re.sub(r'\.{2,}', '.', text)
    return text

# Function to extract sentences containing specific words
def extract_sentences(text, words_to_match):
    # Compile regex pattern to split text into sentences
    sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    
    sentences = sentence_pattern.split(text)
    return [sentence for sentence in sentences if any(word in sentence for word in words_to_match)]

def remove_unnecessary_spaces(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_starting_month(text):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    pattern = r'^(' + '|'.join(months) + r')\s'
    text = re.sub(pattern, '', text)
    return text

def remove_special_characters(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return cleaned_text

def compare_list(list1, list2, threshold=0.7):
    # Preprocess both lists
    list1 = preprocess_list(list1)
    list2 = preprocess_list(list2)
    
    # Check if elements from list1 are similar to elements in list2
    matches = []
    for sentence1 in list1:
        for sentence2 in list2:
            distance = fuzz.ratio(sentence1, sentence2) / 100 
            if distance >= threshold or words_are_subset(sentence1, sentence2):
                match_found = True
                matches.append(match_found)
                break
    return matches

def add_match_column(df, threshold=0.7):
    comparison_results = []

    for index, row in df.iterrows():
        if isinstance(row['Sentences'], list) and isinstance(row['Metaphors Sentence'], list):
            result = compare_list(row['Sentences'], row['Metaphors Sentence'], threshold)
        else:
            result = []
        comparison_results.append(result)
    
    df['Matched_Labeled'] = comparison_results
    df['Matched_Labeled_len'] = df['Matched_Labeled'].apply(len)
    return df

def remove_special_characters_except_colon(text):
    # Define a regex pattern to match all special characters except colon
    pattern = r'[^A-Za-z0-9\s:]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def parse_output(output):
    sentences = []
    categories = []

    # Clean the output text
    cleaned_output = remove_special_characters_except_colon(output)

    # Split by "Sentence:"
    split_output = cleaned_output.split('Sentence:')

    for part in split_output[1:]:  # Skip the first split part which is before the first 'Sentence:'
        if 'Category:' in part:
            # Extract sentence and category
            sentence, category = part.split('Category:')
            sentences.append(sentence.strip().lower())
            category = re.split(r'\s+', category.strip().lower())[0]  # Get the first word for category
            categories.append(category.strip().lower())

    return sentences, categories

def preprocess_sentence(sentence):
    # Remove special characters except for content inside brackets and parentheses
    cleaned_sentence = re.sub(r'[^A-Za-z\s]', '', sentence)
    cleaned_sentence = re.sub(r'\s+', '', cleaned_sentence)  # Remove extra spaces
    return cleaned_sentence.strip().lower()

def is_subsequence(words1, words2):
    it = iter(words2)
    return all(word in it for word in words1)

def words_are_subset(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()
    return is_subsequence(words1, words2) or is_subsequence(words2, words1)

def add_comparison_column(df, threshold=30, name='GPT_1'): 
    comparison_results = []
    category_results = []

    for index, row in df.iterrows():
        if isinstance(row[f'Sentence_{name}'], list) and isinstance(row['Metaphors Sentence'], list):
            result = compare_sentence_lists(row[f'Sentence_{name}'], row['Metaphors Sentence'], threshold)
            result_2 = compare_category_lists(row[f'Category_{name}'], row['Category_Labeled'])
        else:
            result = []
            result_2 = []
        comparison_results.append(result)
        category_results.append(result_2)

    df[f'Comparison_Result_{name}'] = comparison_results
    df[f'Category_Result_{name}'] = category_results
    return df

def add_columns_to_dataframe(df, output, name='GPT'):
    # Initialize lists for the new columns with None or empty lists
    df[f'Sentence_{name}'] = [None] * len(df)
    df[f'Category_{name}'] = [None] * len(df)

    output_index = 0

    for index, row in df.iterrows():
        if len(row['Sentences']) > 0 and output_index < len(output):
            sentences, categories = parse_output(output[output_index])
            df.at[index, f'Sentence_{name}'] = sentences
            df.at[index, f'Category_{name}'] = categories
            output_index += 1
        else:
            df.at[index, f'Sentence_{name}'] = []
            df.at[index, f'Category_{name}'] = []

    return df

def preprocess_list(sentence_list):
    return [preprocess_sentence(sentence) for sentence in sentence_list]

def add_comparison_column_GPT(df, threshold=0.7):
    comparison_results = []

    for index, row in df.iterrows():
        if isinstance(row['Sentence_GPT_Majority'], list) and isinstance(row['Sentences'], list):
            result = compare_sentence_lists_Fuzzy(row['Sentence_GPT_Majority'], row['Sentences'], threshold)
        else:
            result = []
            result_2 = []
        comparison_results.append(result)

    df['Matched_Metaphors'] = comparison_results
    return df

def compare_sentence_lists(list1, list2, threshold=20):
    # Preprocess both lists
    list1 = preprocess_list(list1)
    list2 = preprocess_list(list2)
    
    # Check if elements from list1 are similar to elements in list2
    matches = []
    for sentence1 in list1:
        match_found = False
        for sentence2 in list2:
            distance = Levenshtein.distance(sentence1, sentence2)
            if distance <= threshold or words_are_subset(sentence1, sentence2):
                match_found = True
                break
        matches.append(match_found)
    return matches

def compare_category_lists(list1, list2):
    # Check if elements from list1 are similar to elements in list2
    matches = []
    for category1 in list1:
        match_found = False
        for category2 in list2:
            if category1 == category2:
                match_found = True
                break
        matches.append(match_found)
    return matches

def add_comparison_column_Fuzzy(df, threshold=0.7, name='GPT_1'):
    comparison_results = []
    category_results = []

    for index, row in df.iterrows():
        if isinstance(row[f'Sentence_{name}'], list) and isinstance(row['Metaphors Sentence'], list):
            result = compare_sentence_lists_Fuzzy(row[f'Sentence_{name}'], row['Metaphors Sentence'], threshold)
            result_2 = compare_category_lists(row[f'Category_{name}'], row['Category_Labeled'])
        else:
            result = []
            result_2 = []
        comparison_results.append(result)
        category_results.append(result_2)

    df[f'Matched_Metaphors_Fuzzy_{name}'] = comparison_results
    df[f'Matched_Categories_Fuzzy_{name}'] = category_results
    return df

def compare_sentence_lists_Fuzzy(list1, list2, threshold=0.7):
    # Preprocess both lists
    list1 = preprocess_list(list1)
    list2 = preprocess_list(list2)
    
    # Check if elements from list1 are similar to elements in list2
    matches = []
    for sentence1 in list1:
        match_found = False
        for sentence2 in list2:
            distance = fuzz.ratio(sentence1, sentence2) / 100 
            if distance >= threshold or words_are_subset(sentence1, sentence2):
                match_found = True
                break
        matches.append(match_found)
    return matches

def majority_voting_with_category(df):
    # Function to determine majority voting for a single row
    def get_majority_vote_with_category(row):
        sentences = row['Sentence_GPT_1'] + row['Sentence_GPT_2'] + row['Sentence_GPT_3']
        categories = row['Category_GPT_1'] + row['Category_GPT_2'] + row['Category_GPT_3']
        
        vote_count = pd.Series(sentences).value_counts()
        majority_sentences = vote_count[vote_count >= 2].index.tolist()
        
        majority_categories = []
        for sentence in majority_sentences:
            for i in range(3):
                if sentence in row[f'Sentence_GPT_{i+1}']:
                    idx = row[f'Sentence_GPT_{i+1}'].index(sentence)
                    majority_categories.append(row[f'Category_GPT_{i+1}'][idx])
                    break
        
        return majority_sentences, majority_categories
    
    # Apply the function to each row to get the majority vote and corresponding categories
    df[['Sentence_GPT_Majority', 'Category_GPT_Majority']] = df.apply(lambda row: pd.Series(get_majority_vote_with_category(row)), axis=1)
    
    return df

def calculate_metrics(df_labels, gpt_name, Fuzzy=False, true_metaphor_count=493):
    # Flatten the lists of true metaphors and predicted matches
    total_metaphors = df_labels[f'Sentence_{gpt_name}'].apply(len).sum()
    if Fuzzy:
        true_positive = df_labels[f'Matched_Metaphors_Fuzzy_{gpt_name}'].apply(sum).sum()
        flattened_results = [item for sublist in df_labels[f'Matched_Metaphors_Fuzzy_{gpt_name}'] for item in sublist]
    else:
        true_positive = df_labels[f'Comparison_Result_{gpt_name}'].apply(sum).sum()
        flattened_results = [item for sublist in df_labels[f'Comparison_Result_{gpt_name}'] for item in sublist]

    # Calculate false positives and false negatives
    false_positive = total_metaphors - true_positive
    false_negative = true_metaphor_count - true_positive

    # Print the statistics
    """print(f"Total number of Metaphors (prompt {gpt_name}): {total_metaphors}")
    print(f"Total number of True values for (prompt {gpt_name}): {true_positive}")
    print(f"False Positives: {false_positive}")
    print(f"False Negatives: {false_negative}")"""

    # Calculate precision, recall, and F1 score
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = true_positive / total_metaphors
    
    print(f"Accuracy (prompt {gpt_name}): {accuracy}")
    print(f"Precision (prompt {gpt_name}): {precision}")
    print(f"Recall (prompt {gpt_name}): {recall}")
    print(f"F1 Score (prompt {gpt_name}): {f1}")
    print("")
    return flattened_results

def extract_sentences_column(df):
    all_sentences = []
    
    for index, row in df.iterrows():
        if isinstance(row['Sentences'], list):
            all_sentences.extend(row['Sentences'])
    
    return all_sentences

def extract_metaphors(df, name=None):
    all_metaphors = []
    
    if name is None:
        for index, row in df.iterrows():
            if isinstance(row['Metaphors Sentence'], list):
                all_metaphors.extend(row['Metaphors Sentence'])
    else:
        for index, row in df.iterrows():
            if isinstance(row[f'Sentence_{name}'], list):
                all_metaphors.extend(row[f'Sentence_{name}'])
    
    return all_metaphors

def extract_category(df, name=None):
    all_categories = []
    
    if name is None:
        print("Please provide a name for the column containing the categories")
    else:
        for index, row in df.iterrows():
            if isinstance(row[f'Category_{name}'], list):
                all_categories.extend(row[f'Category_{name}'])
    
    return all_categories

def compare_sentence_fuzzy(sentence1, sentence2, threshold=0.7):
    distance = fuzz.ratio(sentence1, sentence2) / 100
    if distance >= threshold or words_are_subset(sentence1, sentence2):
        return True
    return False

def metrics_dataframe(list_sentences, list_metaphors, list_categories=None):
    # Initialize the dataframe
    matched = []
    category = []
    df_metrics = pd.DataFrame(columns=['Sentence', 'Matched'])
    
    for sentence in list_sentences:
        match_found = False
        category_ = None
        for idx, metaphor in enumerate(list_metaphors):
            if list_categories is not None:
                if compare_sentence_fuzzy(sentence, metaphor):
                    match_found = True
                    category_ = list_categories[idx]
                    break            
            else:
                if compare_sentence_fuzzy(sentence, metaphor, threshold=0.7):
                    match_found = True
                    break
    
        matched.append(match_found)
        if list_categories is not None:
            category.append(category_)

    df_metrics['Sentence'] = list_sentences
    df_metrics['Matched'] = matched

    if list_categories is not None:
        df_metrics['Category'] = category

    return df_metrics

def get_confusion_matrix(df_true, df_pred, model_name='Model'):
    y_true = df_true['Matched']
    y_pred = df_pred['Matched']
    
    """fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show() """ 

    # Calculate and print the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("True Positives (TP):", cm[1, 1])
    print("True Negatives (TN):", cm[0, 0])
    print("False Positives (FP):", cm[0, 1])
    print("False Negatives (FN):", cm[1, 0])       
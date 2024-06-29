import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from textblob import TextBlob
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from ecbdata import ecbdata

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer 
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
warnings.filterwarnings('ignore')

import spacy
sp = spacy.load('en_core_web_sm')

tqdm.pandas()

porter=SnowballStemmer("english")
lmtzr = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

#ECB Color
color = (17/255, 49/255, 147/255)

# Functions:
def strip(word):
    mod_string = re.sub(r'\W+', '', word)
    return mod_string

#the following leaves in place two or more capital letters in a row
#will be ignored when using standard stemming
def abbr_or_lower(word):
    if re.match('([A-Z]+[a-z]*){2,}', word):
        return word
    else:
        return word.lower()

#modular pipeline for stemming, lemmatizing and lowercasing
#note this is NOT lemmatizing using grammar pos 
def tokenize(text, modulation):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences while preserving periods

    processed_sentences = []

    for sentence in sentences:
        if modulation < 2:
            # Tokenize and preprocess each sentence
            tokens = re.findall(r'\w+', sentence.lower())
            stems = []
            for token in tokens:
                lowers = abbr_or_lower(token)
                if lowers not in stop_words and re.search('[a-zA-Z]', lowers):
                    if modulation == 0:
                        stems.append(lowers)
                    elif modulation == 1:
                        stems.append(porter.stem(lowers))
            processed_sentence = " ".join(stems)
        else:
            doc = sp(sentence)
            lemmatized_tokens = []
            for token in doc:
                if token.text.strip():
                    lemmatized_tokens.append(token.lemma_)
            processed_sentence = " ".join(lemmatized_tokens)

        processed_sentences.append(processed_sentence)

    # Reconstruct the text with preserved sentence boundaries
    processed_text = " ".join(processed_sentences)

    return processed_text

def find_metaphors_in_relationships(relationships, as_words, mod=0):
    metaphors = []

    for relationship in relationships:
        # Tokenize the relationship
        relationship = list(relationship)
        relationship = tokenize_word_list(relationship, mod)
        # Check if any word in the tuple is in the as_words list
        if any(word in as_words for word in relationship):
            metaphors.append(relationship)

    return metaphors

def tokenize_word_list(word_list, modulation=0):
    processed_words = []

    for word in word_list:
        processed_word = word.lower()
        if modulation < 2:
            if modulation == 0:
                processed_words.append(processed_word)
            elif modulation == 1:
                processed_words.append(porter.stem(processed_word))
        else:
            # Apply lemmatization using spaCy
            doc = sp(word)
            lemmatized_text=[]
            for w_ in doc:
                lemmatized_text.append(w_.lemma_)
            processed_words.extend([abbr_or_lower(strip(w)) for w in lemmatized_text if (abbr_or_lower(strip(w))) and (abbr_or_lower(strip(w)) not in stop_words)])

    return " ".join(processed_words)

def remove_starting_month(text):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    pattern = r'^(' + '|'.join(months) + r')\s'
    text = re.sub(pattern, '', text)
    return text

def regex_matcher(text, as_words, words_to_match):
    # Split text into sentences using period as delimiter
    sentences = re.split(r'[.!?]\s*', text)

    selected_sentences = []

    for sentence in sentences:
        # Check if the sentence contains any word from words_to_match and any word from as_words
        if any(word in sentence.lower() for word in words_to_match) and any(word in sentence.lower() for word in as_words):
            selected_sentences.append(sentence)

    return selected_sentences

def regex_matcher_word_order(text, as_words, words_to_match):
    # Split text into sentences using period as delimiter
    sentences = re.split(r'[.!?]\s*', text)

    selected_sentences = []

    for sentence in sentences:
        # Check if the sentence contains any word from words_to_match
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in words_to_match):
            # Split the sentence into words
            words = sentence_lower.split()

            # Check if any word from words_to_match exists in the list of words
            match_indices = [i for i, word in enumerate(words) if any(match_word in word for match_word in words_to_match)]

            if match_indices:
                for index in match_indices:
                    # Define the range of words to search around the matched word
                    search_range = range(max(0, index - 3), min(len(words), index + 4))  # Adjusted range to include 3 words before and after

                    search_indices = list(search_range)

                    # Check if any word from as_words appears within the search range
                    if any(words[idx] in as_words for idx in search_indices):
                        selected_sentences.append(sentence)
                        word = [word for word in words_to_match if word in sentence_lower]
                        break  # Break out of loop after finding the first match

    return selected_sentences, word

def vectorize(tokens, vocab):
    vector=[]
    for w in vocab:
        vector.append(tokens.count(w))
    return vector

def text_length_distribution(df):
    df['text_length'] = df['Answers'].apply(len)
    plt.figure(dpi=300)
    # Plot histogram with a label for the legend
    df['text_length'].hist(bins=30, color=color, label='Text Length')
    # Add title and labels
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

def get_length(text):
    if isinstance(text, str):
        return len(text)
    else:
        return 0 
    
def remove_duplicates(sentence):
    words = sentence.split()
    # Convert the list of words into a set to remove duplicates
    unique_words = set(words)
    print(f'Unique words: {len(unique_words)}')
    # Join the unique words back into a string
    result = ' '.join(unique_words)
    return result

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_end_answer(text):
    text = text.replace(r'[end_answer]', '')
    return text

def remove_special_characters_keep_point(text):
    # Define the regex pattern to match special characters except "."
    pattern = r"[^\w\s\.\']"
    # Replace special characters with an empty string
    text = re.sub(pattern, '', text)
    # Replace consecutive dots with just one dot
    text = re.sub(r'\.{2,}', '.', text)
    return text

def remove_unnecessary_spaces(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_relationships(text, words_to_match, lemmatize=False):
    """
    Input - Text
    Output - List of subject-verb, verb-object, adjective-noun relationships extracted from the text
    Caution: Only relationships containing words from the words_to_match list are extracted
    """
    relationships = []
    text = text.lower()
    doc = sp(text)

    # Iterate over each sentence in the text
    for sentence in doc.sents:
        sentence_relationships = []

        # Iterate over each token in the sentence
        for token in sentence:
            # Extract subject-verb relationships (nsubj = nominal subject, csubj = clausal subject)
            if token.dep_ in ['nsubj', 'csubj'] and token.head.pos_ == 'VERB':
                if any(word in [token.text, token.head.text] for word in words_to_match):
                    if lemmatize == False:
                        relationship = (token.text, token.head.text)
                        sentence_relationships.append(relationship)
                    else:
                        relationship = (token.lemma_, token.head.lemma_)
                        sentence_relationships.append(relationship)

            # Extract verb-object relationships (dobj = direct object)
            elif token.dep_ == 'dobj' and token.head.pos_ == 'VERB':
                if any(word in [token.text, token.head.text] for word in words_to_match):
                    if lemmatize == False:
                        relationship = (token.text, token.head.text)
                        sentence_relationships.append(relationship)
                    else:
                        relationship = (token.lemma_, token.head.lemma_)
                        sentence_relationships.append(relationship)

            # Extract adjective-noun relationships (amod = adjectival modifier)
            elif token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
                if any(word in [token.text, token.head.text] for word in words_to_match):
                    if lemmatize == False:
                        relationship = (token.text, token.head.text)
                        sentence_relationships.append(relationship)
                    else:
                        relationship = (token.lemma_, token.head.lemma_)
                        sentence_relationships.append(relationship)

        relationships.extend(sentence_relationships)

    return relationships

def find_duplicates(words):
    duplicates = []
    seen = set()

    for word in words:
        # Check if the word has been seen before
        if word in seen:
            duplicates.append(word)
        else:
            seen.add(word)

    return duplicates

def pos_tagging(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Return part of speech tags
    return blob.tags

def remove_duplicates(sentence):
    words = sentence.split()
    # Convert the list of words into a set to remove duplicates
    unique_words = set(words)
    print(f'Unique words: {len(unique_words)}')
    # Join the unique words back into a string
    result = ' '.join(unique_words)
    return result

def preprocess_text_regex(df, mod):
        text_preproc = (
                df.Answers
                .astype(str)
                .progress_apply(lambda row: remove_end_answer(row))
                .progress_apply(lambda row: remove_numbers(row))
                .progress_apply(lambda row: remove_special_characters_keep_point(row))
                .progress_apply(lambda row: remove_unnecessary_spaces(row))
                .progress_apply(lambda row: remove_starting_month(row))
                .progress_apply(lambda row: tokenize(row, mod)))

        df["Answers_cleaned"]=text_preproc

        return df

def preprocess_text_pos(df, words_to_match, lemma = False):
        text_preproc = (
                df.Answers
                .astype(str)
                .progress_apply(lambda row: remove_end_answer(row))
                .progress_apply(lambda row: remove_numbers(row))
                .progress_apply(lambda row: remove_special_characters_keep_point(row))
                .progress_apply(lambda row: remove_unnecessary_spaces(row))
                .progress_apply(lambda row: remove_starting_month(row))
                .progress_apply(lambda row: extract_relationships(row,words_to_match,lemma)))

        df["pos_relationships"]=text_preproc

        return df
<div style="display: flex; justify-content: space-between;">
    <img src="bse_logo.png" alt="BSE Logo" width="200"/>
    <img src="ecb_logo.png" alt="ECB Logo" width="274"/>
</div>

# DSDM24 Thesis: Identifying Inflation Metaphors in ECB Communications

## Overview

This repository contains the code and data for the thesis project: "Identifying Inflation Metaphors from ECB Communications." The goal of this project is to systematically identify and categorize conceptual metaphors related to inflation in ECB board member communications.

## Contributors


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Preprocessing](#preprocessing)
  - [Metaphor Detection](#metaphor-detection)
  - [Modeling](#modeling)
- [Contributors](#contributors)

## Project Structure

ADD THE TREE LIKE STRUCTURE WITH ALL THE RIGHT NAMES FOR THE NOTEBOOKS

## Methodology

### Data Collection

We manually scraped the necessary information from the ECB website for 19 different board members who have been part of the executive board. The code for this section is available in the notebook "Speakers.ipynb".

### Preprocessing

Our preprocessing pipeline includes:
- Lowercasing all words
- Removing special characters and numbers
- Removing stopwords ([NLTK Stopwords](https://www.nltk.org/search.html?q=stopwords))
- Tokenizing the words
- Lemmatazing (for some cases)

### Metaphor Detection

We employ several approaches to detect metaphors:
1. **Regular Expressions (Regex)**: Identifies metaphors based on predefined keywords and patterns.
2. **Part-of-Speech (POS) Tagging**: Identifies syntactic relationships involving the word "inflation" and our set of keywords.
3. **Neural Networks**: Uses advanced neural network models for better word embeddings and deeper semantic understanding.

### Modeling

Our models are evaluated by comparing their detections with a labeled dataset. The ChatGPT API is leveraged to automate the labeling process, enhancing both efficiency and accuracy. Performance metrics are then computed to assess the models.

## Dependencies
SHOULD WE DO SOMETHING ABOUT INSTALLING? REQUIREMENTS?

<div style="display: flex; justify-content: space-between;">
    <img src="bse_logo.png" alt="BSE Logo" width="200"/>
    <img src="ecb_logo.png" alt="ECB Logo" width="274"/>
</div>

# DSDM24 Thesis: Identifying Inflation Metaphors in ECB Communications

## Overview

This repository contains the code and data for the thesis project: "Identifying Inflation Metaphors from ECB Communications." The goal of this project is to systematically identify and categorize conceptual metaphors related to inflation in ECB board member communications.

## Contributors
*BSE DSDM 2023-2024*  
- **Luis F. Alvarez**  
    Email: [alvarezpluisf@gmail.com](mailto:alvarezpluisf@gmail.com)

- **Sebastien Boxho**  
  Email: [sebastien.boxho@bse.eu](mailto:sebastien.boxho@bse.eu)

- **Mathieu Breier**  
  Email: [mathieu.breier@bse.eu](mailto:mathieu.breier@bse.eu)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Notebooks](#methodology)
  - [Data Collection](#data-collection)
  - [Preprocessing](#preprocessing)
  - [Metaphor Detection](#metaphor-detection)
  - [Modeling](#modeling)
- [Contributors](#contributors)

## Project Structure
```bash
.
├── ChatGPT_Labelling.ipynb
├── Data
│   ├── ECB_InterestRates.csv
│   ├── Final_Data.csv
│   ├── Labeled_Data.csv
│   ├── Prediction_Data.csv
│   ├── Scraped_Data.csv
│   └── Speakers_Info.csv
├── ECB_scrapper.ipynb
├── GPT_Output
│   ├── gpt_answer_1.pkl
│   ├── gpt_answer_2.pkl
│   ├── gpt_answer_3.pkl
│   ├── gpt_answer_all_1.pkl
│   ├── gpt_answer_all_2.pkl
│   └── gpt_answer_all_3.pkl
├── GPT_Prompts
│   ├── Prompt_1.txt
│   ├── Prompt_2.txt
│   └── Prompt_3.txt
├── Metaphor_Extraction.ipynb
├── Metaphors_Descriptive_Analysis.ipynb
├── Metaphors_Identification_Analysis.ipynb
├── README.md
├── ReadmeExtras
│   ├── bse_logo.png
│   └── ecb_logo.png
├── Speakers.ipynb
├── __pycache__
│   └── aux_functions.cpython-310.pyc
├── src
│   ├── Metaphor_Detection_Functions.py
│   ├── Metaphor_Labelling_Functions.py
│   ├── Metrics.py
│   └── __pycache__
└── tree.txt

8 directories, 29 files
```


## Notebooks

### Scrapper

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

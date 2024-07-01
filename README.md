<div style="display: flex; justify-content: space-between;">
    <img src="ReadmeExtras/bse_logo.png" alt="BSE Logo" width="200"/>
    <img src="ReadmeExtras/ecb_logo.png" alt="ECB Logo" width="274" align="right"/>
</div>

# Identifying Inflation Metaphors in ECB Communications

## Abstract

This study systematically identifies and categorizes conceptual metaphors related to inflation in ECB board member communications, following the methodology outlined by Chunyu Hu and Zhi Chen in their paper, "Inflation Metaphor in Contemporary American English". Analyzing a corpus of ECB communications from 2005 to 2024, we identify and classify inflation-related metaphors using regular expressions (REGEX), part-of-speech (POS) tagging, and neural network models. Our findings reveal a consistent use of inflation metaphors across ECB executive board members, with notable variations in frequency and type depending on the individual and the time period. We leveraged the GPT API (OpenAI API) to consistently label and categorize these metaphors, enabling robust performance analysis on the models and contributing to the emerging field of research in AI-driven text analysis.

## Repository Overview

This repository contains the code and data for the thesis project: "Identifying Inflation Metaphors from ECB Communications." The goal of this project is to systematically identify and categorize conceptual metaphors related to inflation in ECB board member communications. The whole of the exercise was performed in python, partitioned in different jupyter notebooks and extra functions imported from python files.

## Contributors
*BSE DSDM 2023-2024*  
- **Luis F. Alvarez**  
  Github: [lalvarezpoli](https://github.com/lalvarezpoli)  
  Email: [alvarezpluisf@gmail.com](mailto:alvarezpluisf@gmail.com)

- **Sebastien Boxho**  
  Github: [SBoxho](https://github.com/SBoxho)  
  Email: [sebastien.boxho@bse.eu](mailto:sebastien.boxho@bse.eu)

- **Mathieu Breier**  
  Github: [mtbrr26](https://github.com/mtbrr26)  
  Email: [mathieu.breier@bse.eu](mailto:mathieu.breier@bse.eu)

  

## Table of Contents

- [Abstract](#abstract)
- [Repository Overview](#repository-overview)
- [Contributors](#contributors)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
  - Scrapper
  - Speakers
  - GPT API
  - Metaphors Identification
  - Results
  - Analysis
- [Dependencies](#dependencies)

## Notebooks

### 0) Scrapper
This notebook serves as a pipeline to automate the process of opening the browser, navigating to ecb.europa.eu and scrapp the information of the interviews. Its second instance actually performes the scraping of the content from each interview individually, and separate the answers from the questions.

### 1) Speakers
We manually scraped the necessary information from the ECB website for 19 different board members who have been part of the executive board. 

### 2) GPT API
In this notebook the Chat GPT API will be used to identify and classify metaphors against manual labeled data. We will then compare the perfromance of GPT 4o. Since the perfromance is considered as good enough to our human labeled observations, we will use GPT to label the entire dataset consisting of over 500 interviews. This final labels will work as the benchmark for the models we implement.

### 3) Metaphors Identification
In this Notebook you can find all the methods used to identify and extract conceptual metaphors about inflation. We also put in place a pre-processing pipeline in order to work with text data. Our preprocessing pipeline includes:
- Lowercasing all words
- Removing special characters and numbers
- Removing stopwords ([NLTK Stopwords](https://www.nltk.org/search.html?q=stopwords))
- Tokenizing the words
- Lemmatazing (for some cases)

We employ several approaches to detect metaphors:
1. **Regular Expressions (Regex)**: Identifies metaphors based on predefined keywords and patterns.
2. **Part-of-Speech (POS) Tagging**: Identifies syntactic relationships involving the word "inflation" and our set of keywords.
3. **Neural Networks**: Uses advanced neural network models for better word embeddings and deeper semantic understanding.




### 4) Results
This notebook is meant to analyse the metaphors detected by the REGEX, Part of Speech, Neural Network and Large Language Model approaches. It will provide results from the modelling and an overall comparisson between all the models and the API results.

The analysis will focus on:
- Frequency of the word "inflation" related to the inflation rate
- Amount of Metaphors detected by each model
- Sentiment analysis of the metaphors flagged

### 5) Analysis
This final notebook perfoms a more indepth analysis of the results, making use of the speakers information and the results obtained from the matephors classification into the categories.

## Project Structure
```bash
.
├── 00_ECB_scrapper.ipynb
├── 01_Speakers.ipynb
├── 02_ChatGPT_Labelling.ipynb
├── 03_Metaphor_Extraction.ipynb
├── 04_Metaphors_Identification_Analysis.ipynb
├── 05_Metaphors_Descriptive_Analysis.ipynb
├── Data
│   ├── ECB_InterestRates.csv
│   ├── Final_Data.csv
│   ├── Labeled_Data.csv
│   ├── Prediction_Data.csv
│   ├── Scraped_Data.csv
│   └── Speakers_Info.csv
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
├── README.md
├── ReadmeExtras
│   ├── bse_logo.png
│   ├── ecb_logo.png
│   └── tree.txt
├── src
│   ├── Metaphor_Detection_Functions.py
│   ├── Metaphor_Labelling_Functions.py
│   └── Metrics.py
└── tree.txt

6 directories, 29 files
```

## Dependencies
A list of sufficient the libraries is posted in the 'requirements.py' file. 

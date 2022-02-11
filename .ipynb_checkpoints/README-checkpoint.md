# Week12_Homework
**Natural Language Processing to understand the sentiment in the latest news articles featuring Bitcoin and Ethereum**

## Background

> There's been a lot of hype in the news lately about cryptocurrency, so you want to take stock, so to speak, of the latest news headlines regarding Bitcoin and Ethereum to get a better feel for the current public sentiment around each coin.
In this assignment, you will apply natural language processing to understand the sentiment in the latest news articles featuring Bitcoin and Ethereum. You will also apply fundamental NLP techniques to better understand the other factors involved with the coin prices such as common words and phrases and organizations and entities mentioned in the articles.

> In this assignment we will complete the following tasks:

1. Sentiment Analysis - A contextual mining of text (The news media) which identifies and extracts subjective information in source material, and helping the public to understand the social sentiment of the crypto currencies in question.

2. Natural Language Processing - The application of computational techniques to the analysis and synthesis of natural language and speech. To convert that data into valuable insights—through identifying whether the news about the crypto currencies are positive or negative. Using the popular NLP Python libraries, including NLTK, scikit-learn, spaCy.

3. Named Entity Recognition - Is a process in Machine Learning, is a cutting-edge machine learning system that unlocks the full potential of our textual data by linking it to existing sources of structured knowledge. Trained on millions of business-related documents, NER is the only technology on the market specifically optimized to extract financial entity information from text documents. Using the spaCy Library 



## 1. Sentiment Analysis

> In this section we Use the newsapi to pull the latest news articles for Bitcoin and Ethereum and create a DataFrame of sentiment scores for each coin. Use descriptive statistics to answer the following questions:

- Which coin had the highest mean positive score?
- Which coin had the highest negative score?
- Which coin had the highest positive score?

> First Step we will import the libraries and packages required for our sentiment analysis and NLP

- NLTK : Natural Langauge Toolkit, the leading platform for building Python programs to work with human langauge data.

- Download NLTK ('vader_lexicon') : VADER **Valence Aware Dictionary and sEntiment Reasoner** is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains.
- from nltk.sentiment.vader import SentimentIntensityAnalyzer : NLTK sentiment analysis package, it's used to detect polarity (positive or negative openion) within a text 

- from newsapi import NewsApiClient : using the API tokens acquired by registering the newsAPI

## 2. Natural Langauge Processing

> In this section, you will use NLTK and Python to tokenize the text for each coin.

### Tokenize 

We need to make sure to :

1. Lowercase each word : regex = re.compile("[^a-zA-Z ]")
2. Remove Punctuation : re_clean = regex.sub('', text)
3. Remove Stopwords : words = word_tokenize(re_clean)

> Import dependencies for the NLP - Tokenisation

- from nltk.tokenize import word_tokenize, sent_tokenize : Tokenizers divide strings into lists of substrings, used to find the words and punctuations in a string
- from nltk.corpus import stopwords : The modules in this package provide functions that can be used to read corpus files in a variety of formats.
- from nltk.stem import WordNetLemmatizer, PorterStemmer : NLTK Stemmers are Interfaces used to remove morphological affixes from words, leaving only the word stem. Stemming algorithms aim to remove those affixes required for eg. grammatical role, tense, derivational morphology leaving only the stem of the word. This is a difficult problem due to irregular words (eg. common verbs in English), complicated morphological rules, and part-of-speech and sense ambiguities (eg. ceil- is not the stem of ceiling).
- from string import punctuation
- import re : A reguler expressionspecifies a set of strings that matches it; the functions in this module let us check if a particular string matches a given a regular expression.

### N-grams

> N-grams are contiguous sequences of n-items in a sentence. N can be 1, 2 or any other positive integers, although usually we do not consider very large N because those n-grams rarely appears in many different places.

> In this subsection of NLP, we will look at the ngrams and word frequency for each coin.

- Use NLTK to produce the ngrams for N = 2.
- List the top 10 words for each coin.

### Wordclouds

> Word Clouds (also known as wordle, word collage or tag cloud) are visual representations of words that give greater prominence to words that appear more frequently. 

- Import dependencies for the wordclouds 

- from wordcloud import WordCloud
- import matplotlib.pyplot as plt
- plt.style.use('seaborn-whitegrid')
- import matplotlib as mpl
- mpl.rcParams['figure.figsize'] = [20.0, 10.0]

## 3. Named Entity Recognition

> In this section, you will build a named entity recognition model for both coins and visualize the tags using SpaCy.

- import spacy
- from spacy import displacy
- python -m spacy download en_core_web_sm

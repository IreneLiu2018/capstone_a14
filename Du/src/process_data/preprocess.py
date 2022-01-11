import pandas as pd
import numpy as np
import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pickle
import nltk
nltk.download('wordnet')

YEAR_THRESHOLD = 2015

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess_abstract(text):
    result = []
    redundant = ['abstract', 'purpose', 'paper', 'goal']
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in redundant:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


def get_cleaned_doc_author_info(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data = data.fillna('')

    
    stemmer = PorterStemmer()

    data['abstract_processed'] = data['abstract'].apply(preprocess_abstract)
    
    data['year'] = data['year'].astype(int)
    data = data[data['year'] > YEAR_THRESHOLD]
    

    # organzie author's abstracts by year
    authors = {}
    for author in data['HDSI_author'].unique():
        authors[author] = {
            2016 : list(),
            2017 : list(),
            2018 : list(),
            2019 : list(),
            2020 : list(),
            2021 : list()
        }
    for i, row in data.iterrows():
        authors[row['HDSI_author']][row['year']].append(row['abstract_processed'])

    all_docs = []
    missing_author_years = {author : list() for author in data['HDSI_author'].unique()}
    for author, author_dict in authors.items():
        for year, documents in author_dict.items():
            if len(documents) == 0:
                missing_author_years[author].append(year)
                continue
            all_docs.append(" ".join(documents))

    return all_docs, authors, missing_author_years, data

def save_cleaned_corpus(data_path, output_path_corpus, output_path_authors, output_path_missing_author_years, output_processed_data_path):
    corpus, authors, missing_author_years, processed_data = get_cleaned_doc_author_info(data_path)
    pickle.dump(corpus, open(output_path_corpus, 'wb'))
    pickle.dump(authors, open(output_path_authors, 'wb'))
    pickle.dump(missing_author_years, open(output_path_missing_author_years, 'wb'))
    processed_data.to_csv(output_processed_data_path)
import pandas as pd
import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(123)
import pickle
import nltk
nltk.download('wordnet')

YEAR = 2015

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess_abstract(text):
    result = []
    redundant = ['abstract', 'purpose', 'paper', 'goal', 'usepackage', 'cod']
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in redundant:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


def data_cleaning(data_path):

    # 'data/raw/final_hdsi_faculty_updated.csv'
    print('get cleaned data ready for modeling')

    data = pd.read_csv(data_path)
    # print(data)
    data = data[data['year'] >= YEAR]
    data['abstract'] = data['abstract'].apply(lambda x: '' if type(x) == float else x)

    def standardize_abstract(abstract):
        abstract = abstract.replace('\n', ' ')
        abstract = abstract.replace('  ', ' ')
        abstract = abstract.replace('-', ' ')
        abstract = abstract.replace('.', '')
        abstract = abstract.replace(':', '')
        abstract = abstract.replace(';', '')
        abstract = abstract.replace(',', '')
        abstract = abstract.replace('"', '')
        abstract = abstract.lower()
        return abstract

    def standardize_title(title):
        title = title.replace('\n', ' ')
        title = title.replace('  ', ' ')
        title = title.replace('-', ' ')
        title = title.replace('.', '')
        title = title.replace(':', '')
        title = title.replace(';', '')
        title = title.replace(',', '')
        title = title.replace('"', '')
        title = title.lower()
        return title

    data['year'] = data['year'].astype(int)
    data['abstract'] = [standardize_abstract(text) for text in data['abstract']]
    data['title_standardized'] = [standardize_title(text) for text in data['title']]
    data['abstract_processed'] = data['abstract'].apply(preprocess_abstract)
    data.drop_duplicates(inplace=True, subset=['abstract'])
    data.drop_duplicates(inplace=True, subset=['title_standardized'])
    data.drop_duplicates(inplace=True, subset=['abstract_processed'])
    # data.dropna(axis=0, how='any')
    data.reset_index(inplace=True)
    data.drop(axis=1, labels=['index'], inplace=True)


    authors = {}
    for author in data.HDSI_author.unique():
        authors[author] = {
            2015 : list(),
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
    missing_author_years = {author : list() for author in data.HDSI_author.unique()}
    for author, author_dict in authors.items():
        for year, documents in author_dict.items():
            if len(documents) == 0:
                missing_author_years[author].append(year)
                continue
            all_docs.append(" ".join(documents))

    return data, authors, all_docs, missing_author_years


def save_cleaned_variables(data_path, output_path_df, output_path_authors, output_processed_data_path, output_path_missing_author_years):
    print('dataframe, authors, processed_data (all_docs), missing_author_years saved')
    dataframe, authors, all_docs, missing_author_years = data_cleaning(data_path)
    pickle.dump(dataframe, open(output_path_df, 'wb'))
    pickle.dump(authors, open(output_path_authors, 'wb'))
    pickle.dump(all_docs, open(output_processed_data_path, 'wb'))
    pickle.dump(missing_author_years, open(output_path_missing_author_years, 'wb'))
    


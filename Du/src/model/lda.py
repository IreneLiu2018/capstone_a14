import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
np.random.seed(123)


def train_lda_model(corpus_path, authors_path, missing_author_year_path, k):

    with open(corpus_path, 'rb') as fp:
        all_docs = pickle.load(fp)

    with open(authors_path, 'rb') as fp:
        authors = pickle.load(fp)

    with open(missing_author_year_path, 'rb') as fp:
        missing_author_year = pickle.load(fp)

    # initate LDA model
    countVec = CountVectorizer()
    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()


    # k topics model 
    lda = LatentDirichletAllocation(n_components=k, n_jobs=-1, random_state=123)
    model_lda = lda.fit_transform(counts)

    # column names
    topicnames = ["Topic" + str(i) for i in range(k)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(all_docs))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(model_lda, columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # add author and year
    df_document_topic['author'] = np.nan
    df_document_topic['year'] = np.nan
    df_document_topic.shape

  
    year_paper_count = {}

    for author, author_dict in authors.items():
        if author not in year_paper_count.keys():
            year_paper_count[author] = 0

        for year, documents in author_dict.items():
            if len(documents) == 0:
                continue
            year_paper_count[author] += 1


    def flatten(A):
        rt = []
        for i in A:
            if isinstance(i,list): rt.extend(flatten(i))
            else: rt.append(i)
        return rt
        
    author_list = list(year_paper_count.keys())
    author_list_populate = [[a]*year_paper_count[a] for a in year_paper_count]
    author_list_populate = flatten(author_list_populate)
    df_document_topic.iloc[:, k+1] = author_list_populate

    year = []
    for a in missing_author_year:
        year.append(list(set([2016, 2017, 2018, 2019, 2020, 2021]) - set(missing_author_year[a])))
    year = flatten(year)
    df_document_topic.iloc[:, k+2] = year

    time_author_topic = df_document_topic

    author_topic = time_author_topic.groupby('author').agg({'dominant_topic': lambda x:x.value_counts().index[0]})

    return model_lda, df_document_topic, time_author_topic, author_topic

def train_lda_5k_dash(corpus_path, authors_path, models_path, results_path, num_topics):

    
    with open(corpus_path, 'rb') as fp:
        all_docs = pickle.load(fp)

    with open(authors_path, 'rb') as fp:
        authors = pickle.load(fp)

    # initate LDA model
    countVec = CountVectorizer()
    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()

    dict_models = {}
    dict_results = {}
    for k in num_topics:
        # k topics model 
        lda = LatentDirichletAllocation(n_components=k, n_jobs=-1, random_state=123)
        model_lda = lda.fit_transform(counts)
        dict_models[str(k)] = lda
        dict_results[str(k)] = model_lda

    pickle.dump(dict_models, open(models_path, 'wb'))
    pickle.dump(dict_results, open(results_path, 'wb'))

    return dict_models, dict_results

def save_lda_model(corpus_path, authors_path, k, model_output_path, dominant_topic_output, missing_author_year_path, time_author_topic_output, author_topic_output):
    model_lda, df_document_topic, time_author_topic, author_topic = train_lda_model(corpus_path, authors_path, missing_author_year_path, k)
    pickle.dump(model_lda, open(model_output_path, 'wb'))
    df_document_topic.to_csv(dominant_topic_output)
    time_author_topic.to_csv(time_author_topic_output)
    author_topic.to_csv(author_topic_output)
    
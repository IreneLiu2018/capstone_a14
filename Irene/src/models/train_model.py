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

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# read feature vector
# all_docs = pickle.load(open('src/features/features.pkl', 'rb'))

def train_and_saved_lda(all_docs_path, num_topics, output_names_path, output_models_path, output_results_path, output_countvec_path):
    countVec = CountVectorizer()

    all_docs = pickle.load(open(all_docs_path, 'rb'))

    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()

    # save models to pkl file

    models = {}
    results = {}
    # num_topics = [10, 15, 20, 25, 30]
    for num_components in num_topics:
        modeller = LatentDirichletAllocation(n_components=num_components, n_jobs=-1, random_state=123)
        result = modeller.fit_transform(counts)
    #     display_topics(modeller, names, 15)
        models[str(num_components)] = modeller
        results[str(num_components)] = result

    print('export models to pkl files')
    # 'src/models/names.pkl'
    pickle.dump(names, open(output_names_path, 'wb'))
    # 'src/models/10_15_20_25_30_models.pkl'
    pickle.dump(models, open(output_models_path, 'wb'))
    # src/models/10_15_20_25_30_results.pkl
    pickle.dump(results, open(output_results_path, 'wb'))
    # 'src/models/10_15_20_25_30_vectorizer.pkl'
    pickle.dump(countVec, open(output_countvec_path, 'wb'))
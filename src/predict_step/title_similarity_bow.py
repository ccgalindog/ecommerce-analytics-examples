import pandas as pd
import os
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from typing import List
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import pickle


def remove_stopwords(xs: str, list_stopwords: List) -> str:
    xs = re.sub(r'[^\w\s]', '', xs)
    ys = ''
    for a_word in xs.split(' '):
        if a_word not in list_stopwords:
            ys = ys + ' ' + a_word
    return ys


def preprocess_text(df_titles: pd.DataFrame) -> pd.DataFrame:
	list_stopwords = stopwords.words('portuguese')

	df_titles['ITE_ITEM_TITLE'] = df_titles['ITE_ITEM_TITLE'].str.lower()

	df_titles['ITE_ITEM_TITLE_PREPROC'] = df_titles['ITE_ITEM_TITLE']\
	                                         .apply(lambda xs:\
	                                           		remove_stopwords(xs,
	                                           					list_stopwords)
	                                           		)
	return df_titles


def apply_tokenization(df_titles):
	with open('models/ex_3_bag_of_words/vectorizer.pkl', 'rb') as handle:
	    vectorizer = pickle.load(handle)

	X_features = vectorizer.transform(df_titles['ITE_ITEM_TITLE_PREPROC'])

	return X_features


def compute_similarity_matrix(X_features):
	all_similarities = np.ones((1,3))
	k = 0
	for i in tqdm(range(X_features.shape[0])):
	    vec_i = X_features[i,:].toarray()
	    vec_j = X_features[i+1:,:].toarray()
	    similarity_index = cosine_similarity(vec_i, vec_j).reshape(-1,1)
	    
	    list_j = np.arange(i+1, X_features.shape[0]).reshape(-1,1)
	    list_i = np.array([i]*len(list_j)).reshape(-1,1)
	    similarity_ij = np.concatenate([list_i, list_j, similarity_index],
	    							   axis=1)
	    all_similarities = np.concatenate([all_similarities, similarity_ij],
	    								  axis=0)

	return all_similarities



def main():

	input_filename = os.getcwd().split('/examples/')[0]\
					 + '/data/initial_data/items_titles_test.csv'

	df_titles = pd.read_csv(input_filename)

	df_titles = preprocess_text(df_titles)

	X_features = apply_tokenization(df_titles)

	all_similarities = compute_similarity_matrix(X_features)

	np.savetxt(f"all_similarities_test.csv", all_similarities, delimiter=",")

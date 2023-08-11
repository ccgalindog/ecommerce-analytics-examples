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
import itertools
import pickle
import tensorflow as tf
import torch
from transformers import BertModel, BertTokenizerFast



def preprocess_text(df_titles: pd.DataFrame) -> pd.DataFrame:

	df_titles['ITE_ITEM_TITLE'] = df_titles['ITE_ITEM_TITLE'].str.lower()

	return df_titles


def get_model(df_titles):

	tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
	model = BertModel.from_pretrained("setu4993/LaBSE")
	model = model.eval()

	return tokenizer, model

def compute_embeddings(df_titles, tokenizer, model):
	i_row = 0
	batch_size = 600
	list_sentences = list(df_titles['ITE_ITEM_TITLE'])
	list_embeddings = list()
	while i_row < df_titles.shape[0]:
		bacth_sentences = list_sentences[i_row : i_row+batch_size]

		tokenized_inputs = tokenizer(bacth_sentences,
									 return_tensors="pt",
								 	 padding=True)

		with torch.no_grad():
			embedding_out = model(**tokenized_inputs)

		list_embeddings.append( embedding_out )
		i_row = i_row + batch_size
	final_embeddings = [each_embedding.pooler_output\
						for each_embedding in list_embeddings]
	final_embeddings_flat = list(itertools.chain(*final_embeddings))
	tensor_embedding = tf.stack(final_embeddings_flat)
	X_features = np.array(tensor_embedding)

	return X_features


def compute_similarity_matrix(X_features):
	similarity_dict = {}
	for i in tqdm(range(df_titles.shape[0])):
	    vec_i = X_features[i:i+1,:]
	    vec_j = X_features[i+1:,:]
	    similarity_dict[i] = cosine_similarity(vec_i, vec_j)

	all_similarities = np.ones((1,3))
	k = 0
	for i, simils in tqdm(similarity_dict.items()):
	    simils = np.array(simils).reshape(-1,1)
	    list_j = np.arange(i+1, len(similarity_dict)+1).reshape(-1,1)
	    list_i = np.array([i]*len(list_j)).reshape(-1,1)
	    similarity_ij = np.concatenate([list_i, list_j, simils], axis=1)
	    all_similarities = np.concatenate([all_similarities,
	    								   similarity_ij],
	    								  axis=0)

	return all_similarities



def main():

	input_filename = os.getcwd().split('/examples/')[0]\
					 + '/data/initial_data/items_titles_test.csv'

	df_titles = pd.read_csv(input_filename)

	df_titles = preprocess_text(df_titles)

	tokenizer, model = get_model(df_titles)

	X_features = compute_embeddings(df_titles, tokenizer, model)

	all_similarities = compute_similarity_matrix(X_features)

	np.savetxt(f"all_similarities_test.csv", all_similarities, delimiter=",")

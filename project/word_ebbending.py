
# Project library
# import project.import_data as import_data
import import_data

# External library
import pandas as pd
# import gensim
import numpy as np
import spacy

nlp = spacy.load('pt_core_news_sm')
# python -m spacy download pt_core_news_sm

def clean_doc(doc):
    text =  [token for token in doc if
            token.text != "pra" and token.text != "para"  and token.text != "e" and token.text != "E" and
            token.text != "" and token.text != " " and token.text != "O" and token.text != "A" and token.text != "a" and token.text != "o" and token.is_punct == False and token.is_stop == False and token.is_bracket == False]
    return text

def clean_to_vectors(doc):
    text_1 = [token for token in doc if
            token.text != "pra" and token.text != "para"  and token.text != "e" and token.text != "E" and
            token.text != "" and token.text != " " and token.text != "O" and token.text != "A" and token.text != "a" and token.text != "o" and token.is_punct == False and token.is_stop == False and token.is_bracket == False]

    text = [token.vector for token in text_1]
    return text

def get_word_vector (dados):
# uso de spacy
    dados['Docs'] = dados['Content'].apply(lambda x: nlp(x)) # comentarios tokenizados pelo spacy
    dados['Docs_clean'] = dados['Docs'].apply(lambda x: clean_doc(x)) # cada linha sao palavras lematizadas, sem pontuacao e stopwords
    dados['Docs_vector'] = dados['Docs'].apply(lambda x: clean_to_vectors(x)) # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario
    # send_dados_to_picle(dados)
    to_cluster_vector = []
    # soma todos os vetores palavras do commentario
    for comment in dados['Docs_vector']:
        for word_vector in comment:
            to_cluster_vector.append(word_vector)

    word_matrix = np.stack(to_cluster_vector, axis=0)
    return word_matrix		


def get_comment_vector (dados):
    # caso 1 - vetor 96

    # uso de spacy
    dados['Docs'] = dados['Content'].apply(lambda x: nlp(x))  # comentarios tokenizados pelo spacy
    # dados['Docs_clean'] = dados['Docs'].apply(lambda x: clean_doc(x))  # cada linha sao palavras lematizadas, sem pontuacao e stopwords
    dados['Comment_vector'] = dados['Docs'].apply(lambda x: comment_to_vector(x))  # cada linha sao vetores das palavras lematizadas, sem pontuacao e stopwords, de cada comentario

    return dados

def similarity_matrix(dados):
	to_cluster_vector = []
	# soma todos os vetores palavras do commentario
	for comment in dados['Comment_vector']:
		sh = comment.shape[0]
		if sh == 96:
			to_cluster_vector.append(comment)
	comment_matrix = np.stack(to_cluster_vector, axis=0)

	similarity_matrix = []
	for i_comment_matrix in range(len(comment_matrix)):
		similarity_matrix.append(list())
		# Preenchimento de 1 vetor vazio por comentários
		for i_comment_matrix_2 in range(len(comment_matrix)):
			comment = []
			cos = np.vdot(comment_matrix[i_comment_matrix], comment_matrix[i_comment_matrix_2])/((np.linalg.norm(comment_matrix[i_comment_matrix]))*np.linalg.norm(comment_matrix[i_comment_matrix_2]))
			similarity_matrix[i_comment_matrix].append(cos)
	comment_matrix = np.stack(similarity_matrix, axis=0)
	return comment_matrix

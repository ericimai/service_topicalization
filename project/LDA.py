# import project.import_data as import_data
# import project.clusterize as cluster
import import_data
import clusterize

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):

    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out



stop_words = stopwords.words('portuguese')

# add new stop words
new_stopwords = ['e']
stop_words.extend(new_stopwords)
#
# data = import_data.import_data_edc()
#
# # data = data.loc[cluster.]
#
# data_words = list(sent_to_words(data['Content'].tolist()))
#
# # print(data_words)
# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(data_words, min_count=1, threshold=1) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=1)
#
# # Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)
#
# # See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
#
# # Remove Stop Words
# data_words_nostops = remove_stopwords(data_words)
#
# # Form Bigrams
# data_words_bigrams = make_bigrams(data_words_nostops)
#
# id2word = corpora.Dictionary(data_words_bigrams)
#
# # Create Corpus
# texts = data_words_bigrams
#
# # Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in texts]
#
# # View
# # print(corpus[:1])
#
#
# # Human readable format of corpus (term-frequency)
# word_freq = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# print(f'word frequency = {word_freq}')
# # Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=5,
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100, # numero de comentarios por training chunk
#                                            passes=10,
#                                            alpha='auto',  # gensim diz que deve ser 1/num_topics
#                                            per_word_topics=True)

# Print the Keyword in the 10 topics
# print(lda_model.print_topics())
# doc_lda = lda_model[corpus]
# for i in lda_model.show_topics(formatted=False,num_topics=lda_model.num_topics,num_words=len(lda_model.id2word)):
#     print (i)

def lda(n_topic,data,tipo ='unigram'):
    data_words = list(sent_to_words(data['Content'].tolist()))

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])


    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    data_words_nostops = lemmatization(data_words_nostops)
    id2word = corpora.Dictionary(data_words_nostops)
    # id2word = corpora.Dictionary(data_words_bigrams)
    bigram = gensim.models.Phrases(data_words, min_count=1, threshold=1)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod )

    if tipo =='bigram':
        texts = data_words_bigrams # usando bigrams
    elif tipo == 'unigram':
        texts = data_words_nostops


    corpus = [id2word.doc2bow(text) for text in texts]


    l = [item for sublist in texts for item in sublist]
    l_corpus = id2word.doc2bow(l)
    # print(l_corpus)
    # Human readable format of corpus (term-frequency)
    word_freq = [(id2word[id], freq) for id, freq in l_corpus]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,

                                               num_topics=n_topic,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100, # numero de comentarios por training chunk
                                               passes=10,
                                               alpha='auto',  # gensim diz que deve ser 1/num_topics
                                               per_word_topics=True)

    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
    return lda_model.print_topics(),word_freq


def lda_from_clusterized_excel(n_topic,excel_input, excel_output):
    '''input: n_topic numero de topicos
        excel_input: excel com os clusters do clusterize
        excel_output: excel com os dados do LDA
    '''
    aux_df = pd.ExcelFile(excel_input)
    # n_sheets = len(aux_df.sheet_names)

    writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
    for sheet_name_cluster in aux_df.sheet_names:
        try:
            # sheet_name_cluster = 'Cluster_' + str(i)
            data = pd.read_excel(excel_input, sheet_name=sheet_name_cluster, index_col=0)
            # preprocessed_data = lemmatization(data)
            lda_topic, w_freq = lda(n_topic, data)
            score_sum = 0
            df_topic = pd.DataFrame(columns=['word', 'score'])
            for word_topic in lda_topic[0][1].split(' + '):
                aux = word_topic.split('*')
                score = float(aux[0])
                word = aux[1].replace('"', '')
                score_sum = score_sum + score
                row = {'word': word, 'score': score}
                df_topic = df_topic.append(row, ignore_index=True)
            df_topic['weighted_score'] = df_topic['score'] / score_sum
            df_topic['weighted_score'] = df_topic['weighted_score'].map("{:.3%}".format)
            df_word_freq = pd.DataFrame(w_freq, columns=['word', 'freq']).sort_values(by=['freq'],ascending=False)
            df_topic = pd.merge(df_topic, df_word_freq, how='inner', left_on='word', right_on='word')
            df_topic.to_excel(writer, sheet_name=sheet_name_cluster)
            df_word_freq.to_excel(writer, sheet_name='word_freq_' + sheet_name_cluster)
        except:
            print('LDA exception\n')
    writer.save()

# def lda_from_clusterized_excel_v2(n_topic,excel_input, excel_output):
#     '''input: n_topic numero de topicos
#         excel_input: excel com os clusters do clusterize
#         excel_output: excel com os dados do LDA
#     '''
#     aux_df = pd.ExcelFile(excel_input)
#     n_sheets = len(aux_df.sheet_names)

#     writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
#     for i in range(0, n_sheets):
#         sheet_name_cluster = 'Cluster_' + str(i)
#         data = pd.read_excel(excel_input, sheet_name=sheet_name_cluster, index_col=0)
#         # preprocessed_data = lemmatization(data)
#         lda_topic, w_freq = lda(n_topic, data)
#         score_sum = 0
#         df_topic = pd.DataFrame(columns=['word', 'score'])
#         for word_topic in lda_topic[0][1].split(' + '):
#             aux = word_topic.split('*')
#             score = float(aux[0])
#             word = aux[1].replace('"', '')
#             score_sum = score_sum + score
#             row = {'word': word, 'score': score}
#             df_topic = df_topic.append(row, ignore_index=True)
#         df_topic['weighted_score'] = df_topic['score'] / score_sum
#         df_topic['weighted_score'] = df_topic['weighted_score'].map("{:.3%}".format)
#         df_word_freq = pd.DataFrame(w_freq, columns=['word', 'freq']).sort_values(by=['freq'],ascending=False)
#         df_topic = pd.merge(df_topic, df_word_freq, how='inner', left_on='word', right_on='word')
#         df_topic.to_excel(writer, sheet_name=sheet_name_cluster)
#         df_word_freq.to_excel(writer, sheet_name='word_freq_' + sheet_name_cluster)
#     writer.save()
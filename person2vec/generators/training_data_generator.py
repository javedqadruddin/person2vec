# python generator yielding training data according to settings
# TODO: implement this with a keras.utils.data_utils.Sequence() to take advantage
# fit_generator(use_multiprocessing=True)

import gensim
from numpy import random.shuffle as rand_shuffle

from person2vec import data_handler

SETTINGS = {'word_vec_source':'../person2vec/data/GoogleNews-vectors-negative300.bin'}


class TrainingDataGenerator(object):
    def __init__(self, word_vec_size, num_compare_entities):
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(SETTINGS['word_vec_source'], binary=True)
        self.handler = data_handler.DataHandler()
        self.word_vec_size = word_vec_size
        self.num_compare_entities = num_compare_entities


    def _get_snippet_index(shuffle):
        index = self.handler.get_snippet_index()
        if shuffle:
            rand_shuffle(index)
        return index


    def flow_from_db(shuffle=True, ):
        snippet_index = get_snippet_index(shuffle)
        for i in snippet_index:

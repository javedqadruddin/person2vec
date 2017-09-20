# python generator yielding training data according to settings
# TODO: implement this with a keras.utils.data_utils.Sequence() to take advantage
# fit_generator(use_multiprocessing=True)

from gensim.models import KeyedVectors
from numpy.random import shuffle as rand_shuffle
from numpy.random import randint as randint
from numpy import zeros as np_zeros
from numpy import array as np_array
from numpy import append as np_append

from person2vec import data_handler

SETTINGS = {'word_vec_source':'../person2vec/data/GoogleNews-vectors-negative300.bin'}


class EmbeddingDataGenerator(object):
    def __init__(self, word_vec_size, num_compare_entities):
        self.word_vectors = KeyedVectors.load_word2vec_format(SETTINGS['word_vec_source'], binary=True)
        self.handler = data_handler.DataHandler()
        self.word_vec_size = word_vec_size
        self.num_compare_entities = num_compare_entities
        self.total_entity_count = self.handler.entity_count()
        self.entity_dict = self._create_entity_dict(self.handler)


    def _create_entity_dict(self, handler):
        entity_dict = {}
        count = 0
        for entity in handler.get_entity_iterator():
            entity_dict.update({entity['name']:count})
            count += 1
        return entity_dict


    def _get_snippet_index(self, shuffle):
        index = self.handler.get_snippet_index()
        if shuffle:
            rand_shuffle(index)
        return index


    # reduce entities to ints to they can be loaded into keras embedding layer
    # create small list of entities as the input to the training
    # make the index of the correct entity name's number for this snippet
    def _create_entity_x_y(self, name):
        entity_num = self.entity_dict[name]
        input_entity_nums = [entity_num]
        for i in range(0, self.num_compare_entities - 1):
            input_entity_nums.append(randint(0, self.total_entity_count))
        rand_shuffle(input_entity_nums)

        # make y the one-hotted index where the correct entity name for this snippet resides
        correct_entity_index = input_entity_nums.index(entity_num)
        one_hot_correct_entity = np_zeros(self.num_compare_entities)
        one_hot_correct_entity[correct_entity_index] = 1
        y = one_hot_correct_entity

        return (input_entity_nums, y)


    def _vectorize_text(self, text):
        vectors = []
        for word in text.split():
            # if there's no word2vec vector for this word, put in a vec of all 0
            try:
                vectors.append(self.word_vectors.word_vec(word))
            except:
                vectors.append(np_zeros(self.word_vec_size))
        return vectors


    # for every snippet in the db, make a training example of the number corre
    # sponding to the person along with n other random person numbers in random
    # order plus the vectorized version of the snippet. y is the position of the
    # correct person in the array of persons
    def flow_from_db(self, shuffle=True, batch_size=32):
        snippet_index = self._get_snippet_index(shuffle)
        batch_x_entity = []
        batch_x_word = []
        batch_y = []
        while True:
            for i in snippet_index:
                snippet = self.handler.get_snippet({'_id':i})
                entity_id = snippet['owner_id']
                entity = self.handler.get_entity({'_id':entity_id})
                entity_x, y = self._create_entity_x_y(entity['name'])
                word_x = self._vectorize_text(snippet['text'])
                batch_x_entity.append(entity_x)
                batch_x_word.append(word_x)
                batch_y.append(y)
                #print(str(batch_y))
                if len(batch_y) >= batch_size:
                    #print('in if statement')
                    yield ([np_array(batch_x_word), np_array(batch_x_entity)], np_array(batch_y))
                    #print('ending batch')
                    batch_x_entity = []
                    batch_x_word = []
                    batch_y = []

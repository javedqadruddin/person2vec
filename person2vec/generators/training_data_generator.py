# python generator yielding training data according to settings
# TODO: implement this with a keras.utils.data_utils.Sequence() to take advantage
# fit_generator(use_multiprocessing=True)

from gensim.models import KeyedVectors
from numpy.random import shuffle as rand_shuffle
from numpy.random import randint as randint
from numpy import zeros as np_zeros
from numpy import array as np_array
from numpy import append as np_append
from os import path

from person2vec import data_handler
from person2vec.utils import preprocessor

HERE = path.abspath(path.dirname(__file__))
PROJECT_DIR = path.dirname(HERE)
DATA_DIR = path.join(PROJECT_DIR, 'data')

SETTINGS = {'word_vec_source':path.join(DATA_DIR, 'GoogleNews-vectors-negative300.bin'),
            'default_db':'person2vec_database'
           }


class EmbeddingDataGenerator(object):
    def __init__(self, word_vec_size, num_compare_entities, db_name=SETTINGS['default_db']):
        self.word_vectors = KeyedVectors.load_word2vec_format(SETTINGS['word_vec_source'], binary=True)
        self.handler = data_handler.DataHandler(db_name)
        self.word_vec_size = word_vec_size
        self.num_compare_entities = num_compare_entities
        self.total_entity_count = self.handler.entity_count()
        self.entity_dict = self._create_entity_dict(self.handler)


    def _create_entity_dict(self, handler):
        entity_dict = {}
        count = 0
        for entity in handler.get_entity_iterator():
            entity_dict.update({entity['_id']:count})
            count += 1
        return entity_dict


    def _get_snippet_index(self, shuffle):
        index = self.handler.get_snippet_index()
        if shuffle:
            rand_shuffle(index)
        return index


    # reduce entities to ints to they can be loaded into keras embedding layer
    # create small list of entities as the input to the training--we want to
    # predict, from a group of entities, which is the one that the snippet's talking about
    # make the index of the correct entity name's number for this snippet
    def _create_entity_x_y(self, id):
        entity_num = self.entity_dict[id]
        input_entity_nums = [entity_num]
        for i in range(0, self.num_compare_entities - 1):
            new_entity_num = entity_num
            # while loop to ensure entities added to training example are not
            # same entity as the correct entity
            while new_entity_num == entity_num:
                new_entity_num = randint(0, self.total_entity_count)
            input_entity_nums.append(new_entity_num)
        # shuffle so that correct entity doesn't always appear in same position
        # without shuffling, net would learn to always just predict correct
        # entity in position 0
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
        # shuffle the order of the snippets
        snippet_index = self._get_snippet_index(shuffle)

        # set up variables to receive the batch of training data
        batch_x_entity = []
        batch_x_word = []
        batch_y = []

        # generators for keras fit_generator need to return data infinitely
        # stopping is handled at a higher level
        while True:
            # for each snippet in our shuffled index
            for i in snippet_index:
                # get the actual snippet using the id
                snippet = self.handler.get_snippet({'_id':i})
                # get the the entity that this snippet corresponds to
                # (this is what we'll be trying to predict)
                entity_id = snippet['owner_id']
                entity = self.handler.get_entity({'_id':entity_id})
                # create an input/output pair on the entity for this snippet
                entity_x, y = self._create_entity_x_y(entity['_id'])
                # remove the entity's name from the snippet if it appears there
                snippet_text = preprocessor.remove_entity_names(snippet['text'], entity['name'])
                # create input from the snippet
                word_x = self._vectorize_text(snippet_text)
                batch_x_entity.append(entity_x)
                batch_x_word.append(word_x)
                batch_y.append(y)
                # when we have enough samples for one batch, yield it
                if len(batch_y) >= batch_size:
                    yield ([np_array(batch_x_word), np_array(batch_x_entity)], np_array(batch_y))
                    # flush the lists when batch is complete
                    batch_x_entity = []
                    batch_x_word = []
                    batch_y = []

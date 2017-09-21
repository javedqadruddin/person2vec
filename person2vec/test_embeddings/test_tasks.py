# a battery of tests to run trained embeddings against
# can also run word2vec embeddings against the same tasks
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from person2vec import data_handler
from person2vec.generators import training_data_generator

TASKS=['gender']


# grabs the embedding from the embedding matrix with index corresponding to num
def _get_entity_vec(num, embeds):
    return embeds[0][num]


# goes faster if a data generator is passed in because initialization steps can then be skipped
def reassociate_embeds_with_names(embeds, data_gen=None):
    # the numbers that stand for entities that were fed to the embedding layer
    # entity_dict contains the entity name to number mapping
    name_and_number = pandas.DataFrame.from_dict(data_gen.entity_dict, orient='index')

    # this gives you a dataframe with 2 columns, name and vector
    name_and_number[0] = name_and_number[0].apply(_get_entity_vec, args=(embeds,))
    name_and_entity_vec = name_and_number
    name_and_entity_vec.columns = ['vector']

    # this gives you dataframe with 1 + number of dimensions of the vector columns
    # 1 column for entity's name and the rest of columns are single values in the vectors
    entity_vecs = name_and_entity_vec.vector.apply(pandas.Series)
    return entity_vecs


def get_embed_weights_from_model(model):
    for i in model.layers:
        if '_embedding' in i.name:
            return i.get_weights()
    raise ValueError("No embedding layer found. Please set the name property of your embedding layer contain the string '_embedding'")


def _name_not_has_vec(name, data_gen):
    try:
        data_gen.word_vectors.word_vec(name.replace(' ','_'))
        return False
    except:
        return True


def run_gender_task(entities, embeds, truncate, data_gen):
    # entities dataframe contains name column as index and gender column as 'male'/'female'
    entities.columns = ['gender']
    # replace male and female with numbers for training
    entities['gender'].replace('female', 0, inplace=True)
    entities['gender'].replace('male', 1, inplace=True)


    # sort them so training input and corresponding outputs will be in same order
    embeds.sort_index(inplace=True)
    entities.sort_index(inplace=True)

    # removes any entities for which there is no word2vec embedding for comparison
    if truncate:
        entities = entities.drop([name for name in entities.index.values if _name_not_has_vec(name, data_gen)])
        embeds = embeds.drop([name for name in embeds.index.values if _name_not_has_vec(name, data_gen)])

    # get the raw y vector for training
    genders = pandas.Series(entities['gender'])
    just_binary_genders = np.array(genders)

    num_train_examples = 750
    train_data = embeds[:num_train_examples].values
    train_labels = just_binary_genders[:num_train_examples]
    test_data = embeds[num_train_examples:].values
    test_labels = just_binary_genders[num_train_examples:]

    # TODO: probably make this a separate model constructor function
    model = Sequential([Dense(1, input_shape=(300,), activation='sigmoid'),])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_data, train_labels,
                verbose=1,
                epochs=90,
                validation_data=(test_data, test_labels))

def _run_tasks(tasks, entities, embeds, truncate, data_gen):
    if 'gender' in tasks:
        run_gender_task(entities.drop(['texts', '_id', 'description', 'occupation'], axis=1), embeds, truncate, data_gen)


def _get_entities_from_db(handler):
    entities = pandas.DataFrame.from_dict(handler.get_all_entities())
    entities.set_index('name', inplace=True)
    return entities


# input a model containing an embedding layer, tests will then be run on the embeddings
# when truncate = True, it will test only on the entities for which word2vec word vectors exist
def test_model(embedding_model, tasks=TASKS, data_gen=None, truncate=True):
    handler = data_handler.DataHandler()

    # can pass a training_data_generator to save time, but, if none is passed, create one
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)


    raw_embeds = get_embed_weights_from_model(embedding_model)
    embeds = reassociate_embeds_with_names(raw_embeds, data_gen)
    entities = _get_entities_from_db(handler)

    _run_tasks(tasks, entities, embeds, truncate, data_gen)


# same as test_model but runs on a set of embeddings passed as an array
def test_embeddings(embeddings, tasks=TASKS, data_gen=None, truncate=True):
    handler = data_handler.DataHandler()

    # can pass a training_data_generator to save time, but, if none is passed, create one
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)

    embeds = reassociate_embeds_with_names(embeddings, data_gen)
    entities = _get_entities_from_db(handler)

    _run_tasks(tasks, entities, embeds, truncate, data_gen)



def _get_word2vec_vector(row):
    return word2vec.word_vec(row.replace(' ','_')).flatten()


def _associate_names_with_word_vecs(entities, data_gen):
    wordvecs_dict = {}
    for name in entities.index:
        twordvecs_dict.update({name:_get_word2vec_vector(name)})
    return pandas.DataFrame.from_dict(wordvecs_dict, orient='index')


def test_word2vec(word2vec_object, tasks=TASKS, data_gen=None):
    handler = data_handler.DataHandler()

    # can pass a training_data_generator to save time, but, if none is passed, create one
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)

    entities = _get_entities_from_db(handler)
    entities = entities.drop([name for name in entities.index.values if _name_not_has_vec(name, data_gen)])
    word_vecs = associate_names_with_word_vecs(entities, data_gen)

    _run_tasks(tasks, entities, word_vecs, truncate=False, data_gen)

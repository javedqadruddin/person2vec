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
def _get_entity_vec(embeds, num):
    return embeds[0][num]


# goes faster if a data generator is passed in because initialization steps can then be skipped
def reassociate_embeds_with_names(embeds, data_gen=None):
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)

    # the numbers that stand for entities that were fed to the embedding layer
    # entity_dict contains the entity name to number mapping
    name_and_number = pandas.DataFrame.from_dict(data_gen.entity_dict, orient='index')

    # this gives you a dataframe with 2 columns, name and vector
    name_and_entity_vec = name_and_number.applymap(_get_entity_vec)
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
        entities = entities.drop([name for name in entities.index.values if _name_not_has_vec(name)])
        embeds = embeds.drop([name for name in embeds.index.values if _name_not_has_vec(name)])

    # get the raw y vector for training
    genders = pandas.Series(entities['gender'])
    just_binary_genders = np.array(genders)

    num_train_examples = int((len(entities) / 4) * 3)
    train_data = embeds[:num_train_examples].values
    train_labels = just_binary_genders[:num_train_examples]
    test_data = embeds[num_train_examples:].values
    test_labels = just_binary_genders[num_train_examples:]

    # TODO: probably make this a separate model constructor function
    model = Sequential([keras.layers.Dense(1, input_shape=(300,), activation='sigmoid'),])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_data, train_labels,
                verbose=1,
                epochs=90,
                validation_data=(test_data, test_labels))

def _run_tasks(tasks, entities, embeds):
    if 'gender' in tasks:
        run_gender_task(entities.drop(['texts', '_id', 'description', 'occupation'], axis=1))


def _get_entities_from_db(handler):
    entities = pd.DataFrame.from_dict(handler.get_all_entities())
    entities.set_index('name', inplace=True)
    return entities


# input a model containing an embedding layer, tests will then be run on the embeddings
# when truncat = True, it will test only on the entities for which word2vec word vectors exist
def test_model(tasks=TASKS, embedding_model, data_gen=None, truncate=True):
    handler = data_handler.DataHandler()

    raw_embeds = get_embed_weights_from_model(embedding_model)
    embeds = reassociate_embeds_with_names(embeds, data_gen)
    entities = _get_entities_from_db(handler)

    _run_tasks(tasks, entities, embeds, truncate, data_gen)


# input a list of embeddings, tests will then be run on them
def test_embeddings(tasks=TASKS, embeds):
    pass

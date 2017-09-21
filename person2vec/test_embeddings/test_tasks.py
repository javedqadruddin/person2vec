# a battery of tests to run trained embeddings against
# can also run word2vec embeddings against the same tasks
import pandas

from person2vec import data_handler
from person2vec.generators import training_data_generator

TASKS=['gender']


# grabs the embedding from the embedding matrix with index corresponding to num
def _get_entity_vec(embeds, num):
    return embeds[0][num]


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


# input a model containing an embedding layer, tests will then be run on the embeddings
def test_model(tasks=TASKS, embedding_model, data_gen=None):
    embeds = get_embed_weights_from_model(embedding_model)

# input a list of embeddings, tests will then be run on them
def test_embeddings(tasks=TASKS, embeddings)

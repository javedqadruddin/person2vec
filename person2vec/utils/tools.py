import pandas


# grabs the embedding from the embedding matrix with index corresponding to num
def _get_entity_vec(num, embeds):
    return embeds[0][num]


# goes faster if a data generator is passed in because initialization steps can then be skipped
def reassociate_embeds_with_ids(embeds, data_gen):
    # the numbers that stand for entities that were fed to the embedding layer
    # entity_dict contains the entity id to number mapping
    id_and_number = pandas.DataFrame.from_dict(data_gen.entity_dict, orient='index')

    # this gives you a dataframe with 2 columns, id and vector
    id_and_number[0] = id_and_number[0].apply(_get_entity_vec, args=(embeds,))

    id_and_entity_vec = id_and_number
    id_and_entity_vec.columns = ['vector']

    # this gives you dataframe with 1 + number of dimensions of the vector columns
    # 1 column for entity's id and the rest of columns are single values in the vectors
    entity_vecs = id_and_entity_vec.vector.apply(pandas.Series)
    return entity_vecs


def get_embed_weights_from_model(model):
    for i in model.layers:
        if '_embedding' in i.name:
            return i.get_weights()
    raise ValueError("No embedding layer found. Please set the name property of your embedding layer contain the string '_embedding'")


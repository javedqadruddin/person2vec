# a battery of tests to run trained embeddings against
# can also run word2vec embeddings against the same tasks
import pandas
import numpy as np
from datetime import date
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from person2vec.utils import tools
from person2vec import data_handler
from person2vec.generators import training_data_generator

TASKS=['gender', 'occupation', 'age', 'political_party']


# grabs the embedding from the embedding matrix with index corresponding to num
def _get_entity_vec(num, embeds):
    return embeds[0][num]


def _name_not_has_vec(name, data_gen):
    try:
        data_gen.word_vectors.word_vec(name.replace(' ','_'))
        return False
    except:
        return True

# removes any entities for which there is no word2vec embedding for comparison
def _truncate_list(entities, data_gen):
    return entities.drop([name for name in entities.index.values if _name_not_has_vec(name, data_gen)])


def _align_frames(entities, embeds):
    # names no longer needed, so set the index to the _id value, which automatically drops the names (which were the index previously)
    if entities.index.name != '_id':
        entities.set_index('_id', inplace=True)

    # removes entries from embeds that are not in entities (this will occur if entities has been truncated)
    embeds = embeds.drop([i for i in embeds.index.values if i not in entities.index])

    # sort them so training input and corresponding outputs will be in same order
    embeds.sort_index(inplace=True)
    entities.sort_index(inplace=True)

    return entities, embeds


def _split_train_test(embeds, labels, num_examples=1000):
    num_train_examples = int(0.75 * num_examples)
    train_data = embeds[:num_train_examples].values
    train_labels = labels[:num_train_examples]
    test_data = embeds[num_train_examples:].values
    test_labels = labels[num_train_examples:]

    return train_data, train_labels, test_data, test_labels


def _to_yrs_since(time_str):
    yr = time_str[1:5]
    month = time_str[6:8]
    day = time_str[9:11]

    try:
        dob = date(int(yr), int(month), int(day))
    except:
        return 'error'

    delta = date.today() - dob

    return delta.days / 365.0


def run_age_task(entities, embeds, truncate, data_gen, embed_size, callbacks):
    print('================TESTING AGE===============================')

    entities.columns = ['_id', 'birth_date']

    # removes any entities for which there is no word2vec embedding for comparison
    if truncate:
        entities = _truncate_list(entities, data_gen)

    entities['age'] = entities['birth_date'].apply(_to_yrs_since)
    entities.drop('birth_date', inplace=True, axis=1)

    # cleaning data with bad date format
    entities = entities[entities.age != 'error']

    entities, embeds = _align_frames(entities, embeds)

    ages = pandas.Series(entities['age'])
    ages_nums = np.array(ages)

    train_data, train_labels, test_data, test_labels = _split_train_test(embeds, ages_nums)

    # TODO: probably make this a separate model constructor function
    model = Sequential([Dense(1, input_shape=(embed_size,), activation='linear'),])
    opt = optimizers.Adam(lr=0.02)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(train_data, train_labels,
                verbose=1,
                epochs=500,
                validation_data=(test_data, test_labels),
                callbacks=callbacks)
    return history


def run_party_task(entities, embeds, truncate, data_gen, embed_size, callbacks):
    print('================TESTING POLITICAL PARTY===============================')

    # entities dataframe contains id column as index and gender column as 'male'/'female'
    entities.columns = ['_id', 'political_party']

    # removes any entities for which there is no word2vec embedding for comparison
    if truncate:
        entities = _truncate_list(entities, data_gen)

    # cleaning data with no party
    entities = entities[entities.political_party != 'unknown']

    entities, embeds = _align_frames(entities, embeds)

    # replace parties with one-hot encoding
    entities = pandas.get_dummies(entities.political_party)

    # get the raw y vector for training
    one_hot_parties = entities.values


    train_data, train_labels, test_data, test_labels = _split_train_test(embeds, one_hot_parties, len(entities))

    # TODO: probably make this a separate model constructor function
    model = Sequential([Dense(len(one_hot_parties[0]), input_shape=(embed_size,), activation='sigmoid'),])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(train_data, train_labels,
                verbose=1,
                epochs=90,
                validation_data=(test_data, test_labels),
                callbacks=callbacks)
    return history


def run_gender_task(entities, embeds, truncate, data_gen, embed_size, callbacks):
    print('================TESTING GENDER===============================')

    # entities dataframe contains id column as index and gender column as 'male'/'female'
    entities.columns = ['_id', 'gender']

    # removes any entities for which there is no word2vec embedding for comparison
    if truncate:
        entities = _truncate_list(entities, data_gen)

    entities, embeds = _align_frames(entities, embeds)

    # replace male and female with numbers for training
    entities['gender'].replace('female', 0, inplace=True)
    entities['gender'].replace('male', 1, inplace=True)

    # get the raw y vector for training
    genders = pandas.Series(entities['gender'])
    just_binary_genders = np.array(genders)

    train_data, train_labels, test_data, test_labels = _split_train_test(embeds, just_binary_genders)

    # TODO: probably make this a separate model constructor function
    model = Sequential([Dense(1, input_shape=(embed_size,), activation='sigmoid'),])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(train_data, train_labels,
                verbose=1,
                epochs=90,
                validation_data=(test_data, test_labels),
                callbacks=callbacks)
    return history


def run_occupation_task(entities, embeds, truncate, data_gen, embed_size, callbacks):
    print('================TESTING OCCUPATION===============================')
    entities.columns = ['_id', 'occupation']

    # removes any entities for which there is no word2vec embedding for comparison
    if truncate:
        entities = _truncate_list(entities, data_gen)

    entities, embeds = _align_frames(entities, embeds)

    # one-hot encode the entities' occupations
    entities = pandas.get_dummies(entities.occupation)

    one_hot_occupations = entities.values

    num_train_examples = 750
    train_data = embeds[:num_train_examples].values
    train_labels = one_hot_occupations[:num_train_examples]
    test_data = embeds[num_train_examples:].values
    test_labels = one_hot_occupations[num_train_examples:]

    # TODO: probably make this a separate model constructor function
    model = Sequential([Dense(4, input_shape=(embed_size,), activation='softmax'),])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_labels,
                verbose=1,
                epochs=90,
                validation_data=(test_data, test_labels),
                callbacks=callbacks)
    return history


def _multi_hot(row, category_index, length):
    out = np.zeros(length)
    for category in row:
        out[category_index[category]] = 1.
    return out


def _get_top_categories(rows):
    category_tally_list = []
    for row in rows:
        for category in row:
            category_tally_list.append(category)
    category_tally_series = pandas.Series(category_tally_list)
    return list(category_tally_series.value_counts()[:10].index)


def _scrub_non_top(categories, top_categories):
    return [category for category in categories if category in top_categories]


def _get_rid_of_small_categories(entities):
    rows = [row for row in entities.categories]
    top_categories = _get_top_categories(rows)
    entities.categories = entities.categories.apply(_scrub_non_top, args=([top_categories]))
    # puts nan into any row with an empty list in category column
   # entities.categories = entities.categories[entities.categories.apply(len) > 0]
    # drops all rows that have a nan
   # entities.dropna(axis=0, inplace=True)
    return entities


def _convert_categories(entities):
    entities = _get_rid_of_small_categories(entities)
    rows = [row for row in entities.categories]
    category_list = []
    for row in rows:
        for category in row:
            if category not in category_list:
                category_list.append(category)

    category_index = {}
    count = 0
    for category in category_list:
        category_index.update({category:count})
        count += 1

    num_categories = len(category_list)

    entities.categories = entities.categories.apply(_multi_hot, args=(category_index, num_categories,))
    labels_series = entities.categories.apply(pandas.Series)

    return labels_series, category_list


def run_biz_type_task(entities, embeds, data_gen, embed_size, callbacks):
    print('================TESTING BUSINESS CATEGORY===============================')
    entities.columns = ['categories']

    entities, category_list = _convert_categories(entities)
    entities, embeds = _align_frames(entities, embeds)


    labels = entities.values

    train_data, train_labels, test_data, test_labels = _split_train_test(embeds, labels, len(labels))


    # num_train_examples = 2300
    # train_data = embeds[:num_train_examples].values
    # train_labels = labels[:num_train_examples]
    # test_data = embeds[num_train_examples:].values
    # test_labels = labels[num_train_examples:]

    # TODO: probably make this a separate model constructor function
    model = Sequential([Dense(len(category_list), input_shape=(embed_size,), activation='sigmoid'),])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels,
                verbose=1,
                epochs=90,
                validation_data=(test_data, test_labels),
                callbacks=callbacks)

    return model, category_list, len(train_data)

def _run_tasks(tasks, entities, embeds, truncate, data_gen, embed_size, callbacks):
    histories = {}
    if 'gender' in tasks:
        to_drop = list(set(entities.columns) - set(['name','_id','gender']))
        histories['gender'] = run_gender_task(entities.drop(to_drop, axis=1), embeds, truncate, data_gen, embed_size, callbacks)
    if 'occupation' in tasks:
        to_drop = list(set(entities.columns) - set(['name','_id','occupation']))
        histories['occupation'] = run_occupation_task(entities.drop(to_drop, axis=1), embeds, truncate, data_gen, embed_size, callbacks)
    if 'age' in tasks:
        to_drop = list(set(entities.columns) - set(['name','_id','birth_date']))
        histories['age'] = run_age_task(entities.drop(to_drop, axis=1), embeds, truncate, data_gen, embed_size, callbacks)
    if 'political_party' in tasks:
        to_drop = list(set(entities.columns) - set(['name','_id','political_party']))
        histories['political_party'] = run_party_task(entities.drop(to_drop, axis=1), embeds, truncate, data_gen, embed_size, callbacks)
    if 'biz_type' in tasks:
        to_drop = list(set(entities.columns) - set(['_id','categories']))
        return run_biz_type_task(entities.drop(to_drop, axis=1), embeds, data_gen, embed_size, callbacks)
    return histories

def _get_entities_from_db(handler, index='name'):
    entities = pandas.DataFrame.from_dict(handler.get_all_entities())
    entities.set_index(index, inplace=True)
    return entities


# input a model containing an embedding layer, tests will then be run on the embeddings
# when truncate = True, it will test only on the entities for which word2vec word vectors exist
def test_model(embedding_model, tasks=TASKS, data_gen=None, truncate=True, embed_size=300, db='person2vec_database', callbacks=[]):
    handler = data_handler.DataHandler(db)

    # can pass a training_data_generator to save time, but, if none is passed, create one
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)


    raw_embeds = tools.get_embed_weights_from_model(embedding_model)
    embeds = tools.reassociate_embeds_with_ids(raw_embeds, data_gen)

    if 'biz_type' in tasks:
        entities = _get_entities_from_db(handler, '_id')
    else:
        entities = _get_entities_from_db(handler)

    return _run_tasks(tasks, entities, embeds, truncate, data_gen, embed_size, callbacks)


# same as test_model but runs on a set of embeddings passed as an array
def test_embeddings(embeddings, tasks=TASKS, data_gen=None, truncate=True, embed_size=300, db='person2vec_database'):
    handler = data_handler.DataHandler(db)

    # can pass a training_data_generator to save time, but, if none is passed, create one
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)

    embeds = tools.reassociate_embeds_with_ids(embeddings, data_gen)
    if 'biz_type' in tasks:
        entities = _get_entities_from_db(handler, '_id')
    else:
        entities = _get_entities_from_db(handler)

    _run_tasks(tasks, entities, embeds, truncate, data_gen)



def _get_word2vec_vector(row, data_gen):
    return data_gen.word_vectors.word_vec(row.replace(' ','_')).flatten()


def _associate_names_with_word_vecs(entities, data_gen):
    wordvecs_dict = {}
    for name in entities.index:
        wordvecs_dict.update({name:_get_word2vec_vector(name,data_gen)})
    return pandas.DataFrame.from_dict(wordvecs_dict, orient='index')

def _get_id_for_name(name, handler):
    return handler.get_entity({'name':name})['_id']


def test_word2vec(word2vec_object, tasks=TASKS, data_gen=None, embed_size=300):
    handler = data_handler.DataHandler()

    # can pass a training_data_generator to save time, but, if none is passed, create one
    if not data_gen:
        data_gen = training_data_generator.EmbeddingDataGenerator(300, 4)

    entities = _get_entities_from_db(handler)
    entities = entities.drop([name for name in entities.index.values if _name_not_has_vec(name, data_gen)])
    word_vecs = _associate_names_with_word_vecs(entities, data_gen)
    word_vecs.reset_index(inplace=True)
    word_vecs['_id'] = pandas.Series([_get_id_for_name(name, handler) for name in word_vecs['index']])
    word_vecs.set_index('index', inplace=True)
    word_vecs.set_index('_id', inplace=True)

    _run_tasks(tasks=tasks, entities=entities, embeds=word_vecs, data_gen=data_gen, truncate=False, embed_size=embed_size)

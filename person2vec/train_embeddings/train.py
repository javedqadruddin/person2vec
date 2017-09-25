from keras.layers import Input, Dense, Embedding, Flatten, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import optimizers
from person2vec import data_handler
from person2vec.generators import training_data_generator


DEFAULT_SETTINGS = {'word_vec_size':300,
                    'embedding_size':300,
                    'num_compare_entities':4,
                    'optimizer':optimizers.adam(),
                    'loss':'categorical_crossentropy',
                    'snippet_size':32
                    }


def _build_default_model(num_compare_entities=DEFAULT_SETTINGS['num_compare_entities'],
                        word_vec_size=DEFAULT_SETTINGS['word_vec_size']):
    # setting variables for size of incoming data
    handler = data_handler.DataHandler()
    num_total_entities = handler.entity_count()
    word_vec_size = DEFAULT_SETTINGS['word_vec_size']
    snip_size = DEFAULT_SETTINGS['snippet_size']
    embedding_size = DEFAULT_SETTINGS['embedding_size']

    input_tensor_words = Input(shape=(snip_size, word_vec_size,), dtype='float32', name='word_input')
    input_tensor_entity = Input(shape=(num_compare_entities,), dtype='int32', name='entity_input')

    word_flatten_layer = Flatten()(input_tensor_words)
    word_dropout_layer = Dropout(0.)(word_flatten_layer)

    entity_embedding_layer = Embedding(num_total_entities, embedding_size, input_length=num_compare_entities, name='entity_embedding')(input_tensor_entity)
    entity_embedding_layer = Flatten()(entity_embedding_layer)
    entity_embedding_layer = Dropout(0.)(entity_embedding_layer)

    word_branch = Dense(1000, activation="relu", name='dense_sentence_layer')(word_dropout_layer)

    joint_embeds = Concatenate(name='joint_embeds')([word_branch, entity_embedding_layer])

    nex = Dropout(0.)(joint_embeds)
    nex = Dense(100, activation="relu", name='dense_consolidator')(nex)
    nex = Dropout(0.)(nex)
    full_out = Dense (4, activation='softmax', name='final_output')(nex)

    model = Model([input_tensor_words, input_tensor_entity], full_out)

    opt = DEFAULT_SETTINGS['optimizer']
    loss = DEFAULT_SETTINGS['loss']
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    return model



# can train a compiled model passed to it, or a default embedding model
def train_model(model=None,
                epochs=100,
                steps_per_epoch=1024,
                embed_size=DEFAULT_SETTINGS['embedding_size'],
                word_vec_size=DEFAULT_SETTINGS['word_vec_size'],
                data_gen= None,
                num_compare_entities=DEFAULT_SETTINGS['num_compare_entities']):

    # if a data_gen is passed in, get variables from it, if not, create a data_gen
    # with default variable values
    if data_gen:
        word_vec_size = data_gen.word_vec_size
        num_compare_entities = data_gen.num_compare_entities
    else:
        data_gen = training_data_generator.EmbeddingDataGenerator(
                                            word_vec_size=word_vec_size,
                                            num_compare_entities=num_compare_entities)

    if not model:
        model = _build_default_model(num_compare_entities, word_vec_size)

    gen = data_gen.flow_from_db()

    model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    # return the model so user can do more with it later, inspect it, etc.
    # return the data_gen because it's frequently needed for testing and takes a while to initialize
    return model, data_gen

# pass name of database as command line argument and this will create an
# embedding for each document in the database, using the 'texts' field of each
# document to create the embeddings. Will then save the embeddings to the
# documents from whence they were created in the 'embed' field

import sys

from person2vec import data_handler
from person2vec.generators import training_data_generator
from person2vec.utils import snippet_creator
from person2vec.train_embeddings import train


def main(db_name):
    handler = data_handler.DataHandler(db_name)
    snippet_creator.snippetize_db(handler)
    data_gen = training_data_generator.EmbeddingDataGenerator(db_name=db_name)
    model, data_gen = train.train_model(data_gen=data_gen)
    handler.save_embeddings_to_db(model, data_gen)


if __name__ == "__main__":
    main(sys.argv)

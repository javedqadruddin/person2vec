from pymongo import MongoClient
from bson.binary import Binary
import pickle
from bson.objectid import ObjectId
from person2vec.utils import tools


SETTINGS = {'default_db': 'person2vec_database'}

class DataHandler(object):
    def __init__(self, db_name=SETTINGS['default_db']):
        client = MongoClient()
        client = MongoClient('localhost', 27017)

        self.db = client[db_name]
        self.entities_collection = self.db.entities
        self.snippets_collection = self.db.snippets

    def _serialize_array_for_mongo(self, array):
        return Binary(pickle.dumps(array, protocol=2), subtype=128)


    def get_snippet_index(self):
        return [snippet['_id'] for snippet in self.get_snippet_iterator()]


    def save_embeddings_to_db(self, model, data_gen, embed_name='embed'):
        embeds = tools.get_embed_weights_from_model(model)
        entity_vecs = tools.reassociate_embeds_as_lists_with_ids(embeds, data_gen)
        for row in entity_vecs.iterrows():
            query = {'_id':ObjectId(row[0])}
            to_store = self._serialize_array_for_mongo(row[1])
            self.update_entity(query, embed_name, to_store)


    # gets embedding for single entity matching the query as a list of floats
    def get_embedding_for_entity(self, query, embed_name='embed'):
        return pickle.loads(self.get_entity(query)[embed_name])


    def create_entity(self, entry):
        try:
            post_id = self.entities_collection.insert_one(entry).inserted_id
            # print("Successfully inserted into db")
            return post_id
        except:
            print("Failed to insert into db")


    def create_snippet(self, entry):
        post_id = self.snippets_collection.insert_one(entry).inserted_id
        return post_id


    def update_entity(self, query, update_field, new_value):
        success = self.entities_collection.update_one(query,
                                            {'$set':{update_field:new_value}},
                                            upsert=False)
        return success.modified_count


    def update_entity_array(self, query, update_field, new_value):
        success = self.entities_collection.update_one(query,
                                            {'$push':{update_field:new_value}},
                                            upsert=False)
        return success.modified_count


    # removes all entities matching query
    def remove_entities(self, query):
        return self.entities_collection.remove(query)


    # removes all snippets matching a query e.g. {owner_id:id}
    def remove_snippets(self, query):
        return self.snippets_collection.remove(query)


    # will return an empty list if no entities match query
    # will return mutiple entities in list if multiple entities match query
    def get_entities(self, query):
        return [entity for entity in self.entities_collection.find(query)]

    # gets a single entity matching the query
    def get_entity(self, query):
        return self.entities_collection.find_one(query)

    def get_snippet(self, query):
        return self.snippets_collection.find_one(query)

    def get_snippets(self, query):
        return [snippet for snippet in self.snippets_collection.find(query)]


    # returns an iterator over all entities matching query
    def get_entity_iterator(self, query=None):
        return self.entities_collection.find(query)


    # returns iterator over all snippets matching query
    def get_snippet_iterator(self, query=None):
        return self.snippets_collection.find(query)


    # returns all entities in the collection
    def get_all_entities(self):
        return [entity for entity in self.entities_collection.find({})]


    def get_all_snippets(self):
        return [snippet for snippet in self.snippets_collection.find({})]



    # returns total number of entities in the collection
    def entity_count(self):
        return self.entities_collection.count()


    # returns total number of snippets in the collection
    def snippet_count(self):
        return self.snippets_collection.count()


    # removes all entities in the collection, returns count of removed entities
    def wipe_entity_collection(self):
        return self.entities_collection.remove({})


    # removes all snippets in the collection, returns count of removed snippets
    def wipe_snippet_collection(self):
        return self.snippets_collection.remove({})

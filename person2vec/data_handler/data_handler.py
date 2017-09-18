from pymongo import MongoClient

class DataHandler(object):
    def __init__(self):
        client = MongoClient()
        client = MongoClient('localhost', 27017)

        self.db = client.person2vec_database
        self.entities_collection = self.db.entities
        self.snippets_collection = self.db.snippets


    def create_entity(self, entry):
        try:
            post_id = self.entities_collection.insert_one(entry).inserted_id
            print("Successfully inserted into db")
            return post_id
        except:
            print("Failed to insert into db")


    def create_snippet(self, entry):
        try:
            post_id = self.snippets_collection.insert_one(entry).inserted_id
            #print("Successfully inserted into db")
            return post_id
        except:
            pass
            #print("Failed to insert into db")


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


    # returns an iterator over all entities matching query
    def get_entity_iterator(self, query=None):
        return self.entities_collection.find(query)


    # returns iterator over all snippets matching query
    def get_snippet_iterator(self, query=None):
        return self.snippets_collection.find(query)


    # returns all entities in the collection
    def get_all_entities(self):
        return [entity for entity in self.entities_collection.find({})]


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

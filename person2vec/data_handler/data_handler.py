from pymongo import MongoClient

class DataHandler(object):
    def __init__(self):
        client = MongoClient()
        client = MongoClient('localhost', 27017)

        self.db = client.person2vec_database
        self.entities_collection = self.db.entities

    def create_entity(self, entry):
        try:
            post_id = self.entities_collection.insert_one(entry).inserted_id
            print("Successfully inserted into db")
            return post_id
        except:
            print("Failed to insert into db")


    # removes all entities matching query
    def remove_entities(self, query):
        removed = self.entities_collection.remove(query)
        return removed


    # will return an empty list if no entities match query
    # will return mutiple entities in list if multiple entities match query
    def get_entities(self, query):
        return [entity for entity in self.entities_collection.find(query)]


    # returns all entities in the collection
    def get_all_entities(self):
        return [entity for entity in self.entities_collection.find({})]


    # returns total number of entities in the collection
    def entity_count(self):
        return self.entities_collection.count()


    # removes all entities in the collection
    def wipe_entity_collection(self):
        removed = self.entities_collection.remove({})
        return removed

# loads the first 10k yelp businesses along with all their reviews and tips into database
import json

from person2vec import data_handler


def load_businesses(handler):
    print('===========Loading Businesses==================')
    with open('../data/yelp/dataset/business.json', 'r') as f:
        count = 0
        for line in f:
            line = line.replace('business_id', '_id')
            line_json = json.loads(line)
            handler.create_entity(line_json)
            handler.update_entity({'_id':line_json['_id']}, 'embed_num', count)
            count += 1
            print('inserted business number ' + str(count))
            if count > 2999:
                break

def load_reviews(business_handler):
    print('===========Loading Reviews==================')
    count = 0
    with open('../data/yelp/dataset/review.json', 'r') as f:
        for line in f:
            line_json = json.loads(line)
            business_id = line_json['business_id']
            text = line_json['text']
            business_handler.update_entity_array({'_id':business_id}, 'texts', text)
            count += 1
            print('inserted review number ' + str(count))


def load_tips(business_handler):
    print('===========Loading Tips==================')
    count = 0
    with open('../data/yelp/dataset/tip.json', 'r') as f:
        for line in f:
            line_json = json.loads(line)
            business_id = line_json['business_id']
            text = line_json['text']
            business_handler.update_entity_array({'_id':business_id}, 'texts', text)
            count += 1
            print('inserted tip number ' + str(count))



def load_yelp_data():
    business_handler = data_handler.DataHandler('yelp_business_database_small')
    load_businesses(business_handler)
    load_reviews(business_handler=business_handler)
    load_tips(business_handler=business_handler)


def main():
    load_yelp_data()

if __name__ == "__main__":
    main()

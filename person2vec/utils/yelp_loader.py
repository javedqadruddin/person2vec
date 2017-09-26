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
            count += 1
            print('inserted business number ' + str(count))


def load_users(handler):
    print('===========Loading Users==================')
    count = 0
    with open('../data/yelp/dataset/user.json', 'r') as f:
        for line in f:
            line = line.replace('user_id', '_id')
            line_json = json.loads(line)
            handler.create_entity(line_json)
            count += 1
            print('inserted user number ' + str(count))


def create_text_entry(entry):
    text = entry['text']
    business_id = entry['business_id']
    user_id = entry['user_id']
    business_entry = {'other_owners':[user_id], 'words':text}
    user_entry = {'other_owners':[business_id], 'words':text}
    return business_entry, user_entry


def load_reviews(business_handler, user_handler):
    print('===========Loading Reviews==================')
    count = 0
    with open('../data/yelp/dataset/review.json', 'r') as f:
        for line in f:
            line_json = json.loads(line)
            business_id = line_json['business_id']
            user_id = line_json['user_id']
            text = line_json['text']
            #business_entry, user_entry = create_text_entry(line_json)
            business_handler.update_entity_array({'_id':business_id}, 'texts', text)
            user_handler.update_entity_array({'_id':user_id}, 'texts', text)
            count += 1
            print('inserted review number ' + str(count))


def load_tips(business_handler, user_handler):
    print('===========Loading Tips==================')
    count = 0
    with open('../data/yelp/dataset/tip.json', 'r') as f:
        for line in f:
            line_json = json.loads(line)
            business_id = line_json['business_id']
            user_id = line_json['user_id']
            text = line_json['text']
            #business_entry, user_entry = create_text_entry(line_json)
            business_handler.update_entity_array({'_id':business_id}, 'texts', text)
            user_handler.update_entity_array({'_id':user_id}, 'texts', text)
            count += 1
            print('inserted tip number ' + str(count))


def load_stars(handler):
    print('===========Loading Stars==================')
    count = 0
    with open('../data/yelp/dataset/review.json', 'r') as f:
        for line in f:
            line_json = json.loads(line)
            user_id = line_json['user_id']
            business_id = line_json['business_id']
            stars = line_json['stars']
            handler.update_entity_array({'_id':user_id}, 'ratings', {business_id:stars})
            count += 1
            print('inserted star number ' + str(count))

def load_yelp_data():
    business_handler = data_handler.DataHandler('yelp_business_database')
    user_handler = data_handler.DataHandler('yelp_user_database')
    load_businesses(business_handler)
    load_users(user_handler)
    load_reviews(business_handler=business_handler, user_handler=user_handler)
    load_tips(business_handler=business_handler, user_handler=user_handler)
    load_stars(user_handler)


def main():
    load_yelp_data()

if __name__ == "__main__":
    main()

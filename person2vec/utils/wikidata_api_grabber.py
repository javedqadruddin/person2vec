from os import path
import requests
import csv
import json
import time

import wiki_extract
from person2vec import data_handler

HERE = path.abspath(path.dirname(__file__))
PROJECT_DIR = path.dirname(HERE)
DATA_DIR = path.join(PROJECT_DIR, 'data')

API_WAIT_TIME = 0.1

WIKIDATA_TITLE_URL = "https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=%s&format=json"
WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles=%s&prop=extracts&explaintext"

# wikidata codes for different entities and relationships
DECODER = {'citizenship':'P27',
            'date of birth':'P569',
            'instance of':'P31',
            'human':'Q5',
            'gender':'P21',
            'Q6581072':'female',
            'female':'Q6581072',
            'Q6581097':'male',
            'male':'Q6581097',
            'occupation':'P106',
            'Q82955':'politician',
            'politician':'Q82955',
            'Q33999':'actor',
            'actor':'Q33999',
            'Q10800557':'film actor',
            'film actor':'Q10800557',
            'Q43845':'businessperson'
            }

HEADERS = {
    'User-Agent': 'ML project for describing famous people',
    'From': 'jqjunk@gmail.com'
}

# instance of = p31
# human = q5
# gender = p21
# male = Q6581097
# female = Q6581072
# citizenship = p27
# date of birth = p569
# occupation = p106
# politician = Q82955
# actor = Q33999
# film actor = Q10800557
# businessperson = Q43845

# https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=Hillary%20Clinton&format=json
# https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&ids=q30&format=json
# https://en.wikipedia.org/w/api.php?action=query&format=json&titles=Johnny_Depp&prop=extracts&explaintext

def _get_wikidata_title(title):
    r = requests.get(WIKIDATA_TITLE_URL % (title), headers=HEADERS)
    return json.loads(r.text)


# takes a type of claim from wikidata taxonomy and pulls that claim type out of
# an entity dict, returning the other end of the claim
# e.g. hillary clinton claim_type='gender' returns 'female'
def _get_claim_entity(person_dict, claim_type):
    return DECODER[person_dict['claims'][DECODER[claim_type]][0]['mainsnak']['datavalue']['value']['id']]


def _get_person_attributes(person_dict):
    attributes_dict = {}
    attributes_dict.update({"description":wiki_extract.get_description(person_dict)})
    attributes_dict.update({"gender":wiki_extract.get_gender(person_dict)})
    attributes_dict.update({"occupation":wiki_extract.get_occupation(person_dict)})
    return attributes_dict


def _check_if_right_person(person_dict, target_name):
    try:
        retrieved_name = wiki_extract.get_title(person_dict)
        description = wiki_extract.get_description(person_dict)
        instance_of = wiki_extract.get_instance_of(person_dict)
        gender = wiki_extract.get_gender(person_dict)
        occ = wiki_extract.get_occupation(person_dict)
    except:
        print("Data in wrong form for " + target_name)
        return False
    if instance_of != 'human':
        print(target_name + " does not appear to be of type human.")
        return False
    elif retrieved_name != target_name:
        print("Wrong person retrieved. Got " + retrieved_name + " instead of " + target_name)
        return False
    return True


def _write_to_csv(rows):
    with open(path.join(DATA_DIR, "people_attributes.csv"), 'wb') as w:
        csv_writer = csv.writer(w)
        for row in rows:
            print(row)
            csv_writer.writerow(row)


def _write_to_db(new_entity):
    handler = data_handler.DataHandler()
    # if entity by that name already exists, remove it
    if handler.get_entities({"name":new_entity["name"]}):
        handler.remove_entities({"name":new_entity["name"]})
    # add the new entity
    handler.create_entity(new_entity)


def _get_person_article(name):
    r = requests.get(WIKIPEDIA_URL % (name), headers=HEADERS)
    article = wiki_extract.get_article(json.loads(r.text))
    return {"texts":[article]}


def main():
    rows_to_write = []
    with open(path.join(DATA_DIR, "people.csv"), 'rb') as people_file:
        people_reader = csv.reader(people_file)

        attempt_counter = 0
        fail_counter = 0
        success_counter = 0

        for row in people_reader:
            attempt_counter += 1
            person_name = row[0]
            #person_occupation = row[1]
            print("Trying " + person_name + " person number: " + str(attempt_counter))
            try:
                person_name = person_name.encode('utf-8')
            except:
                print("Failed utf-8 encoding")
                fail_counter += 1
                continue

            try:
                person_wikidata = _get_wikidata_title(person_name)
                print("Successfully got " + person_name + " from wikidata")
            except:
                print("Failed to get " + person_name + " from wikidata")
                fail_counter += 1
                continue

            try:
                person_article = _get_person_article(person_name)
            except:
                print("Failed to get " + person_name + "'s article from wikipedia")
                fail_counter += 1
                continue

            # get the first entity from the dict that wikidata API returns
            entities_entries = person_wikidata['entities']
            first_entity = entities_entries[entities_entries.keys()[0]]

            # returns True if all attributes successfully extracted
            is_right_person = _check_if_right_person(first_entity, person_name)

            if not is_right_person:
                fail_counter += 1
            else:
                person_attributes = {"name":person_name}
                # adds entries from dict returned by get_person_attributes to the existing dict
                person_attributes.update(_get_person_attributes(first_entity))
                person_attributes.update(person_article)
                # a number that will be this person's number to find its embedding
                person_attributes.update({'num':success_counter})
                _write_to_db(person_attributes)
                success_counter += 1

            # to prevent hitting the wikipedia API too frequently
            #time.sleep(API_WAIT_TIME)

    #_write_to_csv(rows_to_write)

    print(str(attempt_counter) + " attempts made")
    print(str(fail_counter) + " entities failed")
    print(str(success_counter) + " entities succeeded")
    if fail_counter + success_counter == attempt_counter:
        print("Fails and successes added up to attempts.. Good")
    else:
        print("Fails and successes didn't add up to attempts.. something fishy")







if __name__ == "__main__":
    main()

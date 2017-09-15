import json

# wikidata codes for different entities and relationships
DECODER = {'citizenship':'P27',
            'date of birth':'P569',
            'instance_of':'P31',
            'human':'Q5',
            'Q5':'human',
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


def id2word(id):
    return DECODER[id]


def word2id(word):
    return DECODER[word]


def get_instance_of(entity_dict):
    return id2word(entity_dict['claims'][word2id('instance_of')][0]['mainsnak']['datavalue']['value']['id'])


def get_title(entity_dict):
    return entity_dict['labels']['en']['value']


def get_description(entity_dict):
    return entity_dict['descriptions']['en']['value']

def get_gender(entity_dict):
    id2word([entity_dict['claims'][word2id('gender')][0]['mainsnak']['datavalue']['value']['id']])

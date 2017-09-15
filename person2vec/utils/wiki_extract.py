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
            'Q43845':'businessperson',
            'Q639669':'musician',
            'musician':'Q639669',
            'Q177220':'singer',
            'singer':'Q177220'
            }


def id2word(id):
    return DECODER[id]


def word2id(word):
    return DECODER[word]


def get_article(article_dict):
    pages = article_dict['query']['pages']
    return pages[pages.keys()[0]]['extract']


def get_instance_of(entity_dict):
    return id2word(entity_dict['claims'][word2id('instance_of')][0]['mainsnak']['datavalue']['value']['id'])


def get_title(entity_dict):
    return entity_dict['labels']['en']['value']


def get_description(entity_dict):
    return entity_dict['descriptions']['en']['value']


def get_gender(entity_dict):
    return id2word([entity_dict['claims'][word2id('gender')][0]['mainsnak']['datavalue']['value']['id']][0])


def get_occupation(entity_dict):
    occupations = entity_dict['claims'][word2id('occupation')]
    occ_names = []
    for occupation in occupations:
        occ_names.append(occupation['mainsnak']['datavalue']['value']['id'])
    if word2id('politician') in occ_names:
        return 'politician'
    elif word2id('actor') in occ_names or word2id('film actor') in occ_names:
        return 'actor'
    elif word2id('singer') in occ_names or word2id('musician') in occ_names:
        return 'musician'
    else:
        return 'businessperson'

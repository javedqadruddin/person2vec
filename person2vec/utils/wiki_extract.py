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
            'businessperson':'Q43845',
            'Q639669':'musician',
            'musician':'Q639669',
            'Q177220':'singer',
            'singer':'Q177220',
            'date of birth':'P569',
            'member of political party':'P102',
            'democrat':'Q29552',
            'republican':'Q29468'
            }

CLAIMS_LIST = ['P27',               #citizenship
                'P103',             #native language
                'P39',              #position held
                'P166',             #award received
                'P69',              #educated at
                'P172',             #ethnic group
                'P1303',            #instrument
                'P512',             #academic degree
                'P551']             #residence


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


def get_birth_date(entity_dict):
    try:
        return entity_dict['claims'][word2id('date of birth')][0]['mainsnak']['datavalue']['value']['time']
    except:
        return 'unknown'

def get_party(entity_dict):
    try:
        parties = entity_dict['claims'][word2id('member of political party')]
    except:
        return 'unknown'
    for party in parties:
        try:
            party_name = party['mainsnak']['datavalue']['value']['id']
        except:
            continue
        if word2id('democrat') == party_name:
            return 'democrat'
        if word2id('republican') == party_name:
            return 'republican'
    return 'other'



def get_claims(entity_dict):
    claims = {}
    for claim in CLAIMS_LIST:
        try:
            entry_list = entity_dict['claims'][claim]
        except:
            continue
        claims.update({claim:[]})
        for entry in entry_list:
            value = entry['mainsnak']['datavalue']['value']['id']
            claims[claim].append(value)
    return claims


def get_occupation(entity_dict):
    occupations = entity_dict['claims'][word2id('occupation')]
    for occupation in occupations:
        occ_name = occupation['mainsnak']['datavalue']['value']['id']
        if word2id('politician') == occ_name:
            return 'politician'
        if word2id('actor') == occ_name or word2id('film actor') == occ_name:
            return 'actor'
        if word2id('singer') == occ_name or word2id('musician') == occ_name:
            return 'musician'
        if word2id('businessperson') == occ_name:
            return 'businessperson'
    return 'businessperson'

    # # iterate through the occupations collected and return the first
    # for occ in occ_names:
    #     if word2id('politician') in occ_names:
    #         return 'politician'
    #     elif word2id('actor') in occ_names or word2id('film actor') in occ_names:
    #         return 'actor'
    #     elif word2id('singer') in occ_names or word2id('musician') in occ_names:
    #         return 'musician'
    # return 'businessperson'

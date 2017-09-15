import json
import requests

from person2vec.utils import wiki_extract
from person2vec.utils import wikidata_api_grabber




def setUp():
    WIKIDATA_TITLE_URL = "https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=%s&format=json"
    headers = {
        'User-Agent': 'ML project for describing famous people',
        'From': 'jqjunk@gmail.com'
    }
    r = requests.get(WIKIDATA_TITLE_URL % ('Hillary_Clinton'), headers=headers)
    test_input = json.loads(r.text)
    entities_entries = test_input['entities']
    return entities_entries[entities_entries.keys()[0]]


def test_get_instance_of():
    test_entity = setUp()
    assert wiki_extract.get_instance_of(test_entity) == 'human'

def test_get_title():
    test_entity = setUp()
    assert wiki_extract.get_title(test_entity) == 'Hillary Clinton'

def test_get_description():
    test_entity = setUp()
    assert wiki_extract.get_description(test_entity) == 'American politician, senator, and U.S. Secretary of State'

def test_get_gender():
    test_entity = setUp()
    assert wiki_extract.get_gender(test_entity) == 'female'

def test_get_person_attributes():
    test_entity = setUp()
    assert wikidata_api_grabber._get_person_attributes(test_entity) == {"description":"American politician, senator, and U.S. Secretary of State",
                                                                        "gender":"female",
                                                                        "occupation":"politician"}
def test_get_occupation():
    test_entity = setUp()
    assert wiki_extract.get_occupation(test_entity) == 'politician'

# def test_get_article():
#     r = requests.get(WIKIDATA_TITLE_URL % ('Johnny_Depp'))
#     assert wiki_extract.get_article(json.loads(r.text)) ==

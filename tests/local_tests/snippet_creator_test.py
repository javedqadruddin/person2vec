from person2vec.utils import snippet_creator
from person2vec import data_handler

SNIPPET_TEST_STR_SIZE = 16


def test_remove_punctuation():
    test_str = "Hello. #$%This (is) a test-string! 99"
    assert snippet_creator.remove_punctuation(test_str) == "Hello This is a test string 99"



def test_slice_into_snippets_start():
    test_str = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'
    result = snippet_creator.slice_into_snippets(test_str, 4, 2)
    assert (result[0] == '1 2 3 4' and
            result[1] == '3 4 5 6' and
            result[2] == '5 6 7 8')

def test_slice_into_snippets_easy_end():
    test_str = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'
    result = snippet_creator.slice_into_snippets(test_str, 4, 2)
    assert (result[-1] == '13 14 15 16' and
            result[-2] == '11 12 13 14')

def test_slice_into_snippets_hard_end():
    test_str = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15'
    result = snippet_creator.slice_into_snippets(test_str, 4, 2)
    assert (result[-1] == '12 13 14 15' and
            result[-2] == '11 12 13 14')

def test_process_text():
    test_str = '1 2 3 4, 5 6 (7) 8.'
    settings = {'snippet_len':4, 'stride':2}
    result = snippet_creator.process_text(test_str, settings)
    assert (result[0] == '1 2 3 4' and
            result[1] == '3 4 5 6' and
            result[2] == '5 6 7 8' and
            len(result) == 3)

def test_get_entity_snippets():
    test_entity = {'description': '43rd President of the United States',
                    'gender': 'male',
                    'name': 'George W. Bush',
                    'occupation': 'politician',
                    'texts':['hello! this is a test text@!',
                            'hi there!']}
    settings = {'snippet_len':4, 'stride':2}
    result = snippet_creator.get_entity_snippets(test_entity, settings)
    assert result == ['hello this is a',
                        'is a test text',
                        'hi there']

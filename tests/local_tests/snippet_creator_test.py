from person2vec.utils import snippet_creator
from person2vec import data_handler


def test_remove_punctuation():
    test_str = "Hello. #$%This (is) a test-string! 99"
    assert snippet_creator.remove_punctuation(test_str) == "Hello This is a test string 99"


def test_build_snippet_from_repeats():
    test_str = '1 2 3'
    test_words = test_str.split()
    result = snippet_creator._build_snippet_from_repeats(test_words, 5)
    assert result == '1 2 3 1 2'


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
    test_str = '1 2 3 4 5 6 7 8'
    settings = {'snippet_len':4, 'stride':2}
    result = snippet_creator.process_text(test_str, settings, 2)
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
    result = snippet_creator.get_entity_snippets(test_entity, 3, settings)
    assert result == ['hello this is a',
                        'is a test text',
                        'test text hi there']

def test_get_entity_snippets_short():
    test_entity = {'description': '43rd President of the United States',
                    'gender': 'male',
                    'name': 'George W. Bush',
                    'occupation': 'politician',
                    'texts':['hi there!']}
    settings = {'snippet_len':4, 'stride':2}
    result = snippet_creator.get_entity_snippets(test_entity, 1, settings)
    assert result == ['hi there hi there']



def test_get_entity_snippets_high_biggest_entity():
    test_entity = {'description': '43rd President of the United States',
                    'gender': 'male',
                    'name': 'George W. Bush',
                    'occupation': 'politician',
                    'texts':['hello! this is a test text@!',
                            'hi there!']}
    settings = {'snippet_len':4, 'stride':2}
    result = snippet_creator.get_entity_snippets(test_entity, 10, settings)
    assert result == ['hello this is a',
                    'this is a test',
                    'is a test text',
                    'a test text hi',
                    'test text hi there',
                    'hello this is a',
                    'this is a test',
                    'is a test text',
                    'a test text hi',
                    'test text hi there']



def test_get_entity_snippets_high_biggest_entity_truncate():
    test_entity = {'description': '43rd President of the United States',
                    'gender': 'male',
                    'name': 'George W. Bush',
                    'occupation': 'politician',
                    'texts':['hello! this is a test text@!',
                            'hi there!']}
    settings = {'snippet_len':4, 'stride':2}
    result = snippet_creator.get_entity_snippets(test_entity, 9, settings)
    assert result == ['hello this is a',
                    'this is a test',
                    'is a test text',
                    'a test text hi',
                    'test text hi there',
                    'hello this is a',
                    'this is a test',
                    'is a test text',
                    'a test text hi']




def test_concat_all_texts():
    test_list = ['hello my name is alice',
                'I followed a white rabbit',
                'way down the rabbit hole',
                'and therein, I found myself']
    assert snippet_creator.concat_all_texts(test_list) == 'hello my name is alice I followed a white rabbit way down the rabbit hole and therein, I found myself'


def test_get_max_snippets_short():
    assert snippet_creator.get_max_snippets(20, 16) == 1

def test_get_max_snippets_even():
    assert snippet_creator.get_max_snippets(64, 16) == 4

def test_get_max_snippets_odd():
    assert snippet_creator.get_max_snippets(55,16) == 3

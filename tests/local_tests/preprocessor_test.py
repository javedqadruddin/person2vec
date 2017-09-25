# unit tests for preprocessor.py

from person2vec.utils import preprocessor

def test_split_names_hard():
    name = 'George H W Bush'
    assert preprocessor.split_names(name) == ('George', 'Bush', ['H', 'W'])

def test_split_names_easy():
    name = 'Bill Clinton'
    assert preprocessor.split_names(name) == ('Bill', 'Clinton', [])

def test_remove_entity_names_hard():
    name = 'George H W Bush'
    text = 'George H W Bush was president of the H United States before Bill Clinton'
    assert preprocessor.remove_entity_names(text, name) == 'name name name name was president of the H United States before Bill Clinton'

def test_remove_entity_names_easy():
    name = 'Bill Clinton'
    text = 'George H W Bush was president of the H United States before Bill Clinton'
    assert preprocessor.remove_entity_names(text, name) == 'George H W Bush was president of the H United States before name name'

def test_dont_remove_name_part_of_word():
    name = 'George H W Bush'
    text = 'Hello there Wild West'
    assert preprocessor.remove_entity_names(text, name) == 'Hello there Wild West'

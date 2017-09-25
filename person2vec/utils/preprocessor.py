# pre-processes data in the database by removing references to a person's name
# from the snippets about that person and vectorizing the snippets with word2vec
import re

def split_names(name):
    names = name.split()
    first_name = names[0]
    last_name = names[-1]
    middle_names = names[1:-1]
    return (first_name, last_name, middle_names)



# takes two strings, the snippet text and the entity name (first and last in
# same string) and outputs the same text with the names replaced with 'name'
def remove_entity_names(text, name):
    first_name, last_name, middle_names = split_names(name)

    text = re.sub(r"\b%s\b" % first_name, 'name', text)
    text = re.sub(r"\b%s\b" % last_name, 'name', text)
    text = re.sub(r"\b%s\b" % first_name + '_' + last_name, 'name', text)

    if len(middle_names) > 0:
        for mid_name in middle_names:
            text = text.replace('name ' + mid_name, 'name name')
            text = text.replace(mid_name + ' name', 'name name')

    return text

# takes large text documents and creates snippets, stores them in db
from re import compile, sub, UNICODE

# will get a 32 word-long snippet every 16 words, half of each snippet
# overlaps with the previous snippet and half with the next snippet
SETTINGS = {'snippet_len':32, 'stride':16}


# removes punctuation from text and makes unicode-safe at same time
def remove_punctuation(text):
    text = text.replace('-',' ')
    just_alnum_pattern = compile('([^\s\w]|_)+', UNICODE)
    return just_alnum_pattern.sub('', text)


# iterates through list of words until it gets within snippet_len of the end
# once it reaches that point, gives the last snippet_len words as last snippet_len
# and returns
def slice_into_snippets(text, snippet_len, sample_spacing):
    words = text.split()
    subs = []
    for i in range(0, len(words), sample_spacing):
        # avoids having a weird length snippet at the end
        if len(words) - i > snippet_len:
            subs.append(" ".join(words[i: i + snippet_len]))
        else:
            subs.append(" ".join(words[-snippet_len:]))
            break
    return subs


def process_text(text, settings):
    text = remove_punctuation(text)
    snippets = slice_into_snippets(text, settings['snippet_len'], settings['stride'])
    return snippets


# there can be multiple texts saved in the db for each person
# this function snippetizes each text in turn and returns a list of snippets
# from all the texts
def process_texts(texts, settings):
    snippet_list = []
    for text in texts:
        snippets = process_text(text, settings)
        for snippet in snippets:
            snippet_list.append(snippet)
    return snippet_list


def get_entity_snippets(entity, settings):
    return process_texts(entity['texts'], settings)


def write_snippets(handler, entity, snippets):
    for snippet in snippets:
        handler.create_snippet({'owner_id':entity['_id'],
                                'text':snippet})


def snippetize_db(handler):
    num_entities = handler.entity_count()
    count = 0
    for entity in handler.get_entity_iterator():
        count += 1
        snippets = get_entity_snippets(entity, SETTINGS)
        print("writing snippets for " + entity['name'] + " number " + str(count) + " of " + str(num_entities))
        write_snippets(handler, entity, snippets)

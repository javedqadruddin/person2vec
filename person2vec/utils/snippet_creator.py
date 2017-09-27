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


def _build_snippet_from_repeats(words, snippet_len):
    #print(len(words))
    while len(words) < snippet_len:
        words = words + words
    # if it went over the snippet_len, truncate it at snippet_len
    return " ".join(words[:snippet_len])


# iterates through list of words until it gets within snippet_len of the end
# once it reaches that point, gives the last snippet_len words as last snippet_len
# and returns
def slice_into_snippets(text, snippet_len, sample_spacing):
    words = text.split()

    # if this text is smaller than snippet_len, make a snippet of snippet_len
    # that is the text repeated as many times as is necessary to reach snippet_len
    if len(words) < snippet_len:
        return [_build_snippet_from_repeats(words, snippet_len)]

    # subs will be a list of strings. each string will be of length snippet_len
    subs = []
    for i in range(0, len(words), sample_spacing):
        if len(words) - i > snippet_len:
            subs.append(" ".join(words[i: i + snippet_len]))
        # avoids having a weird length snippet at the end
        else:
            subs.append(" ".join(words[-snippet_len:]))
            break
    return subs


def process_text(text, settings):
    snippets = slice_into_snippets(text, settings['snippet_len'], settings['stride'])
    return snippets


# there can be multiple texts saved in the db for each person
# this function snippetizes each text in turn and returns a list of snippets
# from all the texts
def process_texts(texts, settings):
    snippet_list = []
    for text in texts:
        text = remove_punctuation(text)
        if len(text.split()) > 0:
            snippets = process_text(text, settings)
            for snippet in snippets:
                snippet_list.append(snippet)
    return snippet_list


def get_entity_snippets(entity, settings):
    return process_texts(entity['texts'], settings)


def write_snippets(handler, entity, snippets):
    for snippet in snippets:
        # if statement as final check to make sure we don't insert an empty string
        # as a snippet
        if len(snippet) > 0:
            handler.create_snippet({'owner_id':entity['_id'],
                                    'text':snippet})


def snippetize_db(handler):
    num_entities = handler.entity_count()
    count = 0
    for entity in handler.get_entity_iterator():
        # if to make sure there actually are some texts for this entity
        if len(entity['texts']) > 0:
            count += 1
            snippets = get_entity_snippets(entity, SETTINGS)
            print("writing snippets for " + entity['_id'] + " number " + str(count) + " of " + str(num_entities))
            write_snippets(handler, entity, snippets)

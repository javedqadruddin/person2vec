# takes large text documents and creates snippets, stores them in db
from re import compile, sub, UNICODE

# will get a 32 word-long snippet every 16 words, half of each snippet
# overlaps with the previous snippet and half with the next snippet
SETTINGS = {'snippet_len':32, 'stride':16, 'max_expand':2000}

def get_texts_length(texts):
    text_length = 0
    for text in texts:
        text_length += len(text.split())
    return text_length

def get_longest_texts(handler):
    top_length = 0
    for entity in handler.get_entity_iterator():
        text_length = get_texts_length(entity['texts'])
        if text_length > top_length:
            top_length = text_length
    return top_length


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


def process_text(text, settings, stride):
    snippets = slice_into_snippets(text, settings['snippet_len'], stride)
    return snippets


def get_max_snippets(longest_texts_length, max_stride):
    return min(SETTINGS['max_expand'],int(longest_texts_length / max_stride))


def concat_all_texts(texts):
    return " ".join(texts)


# figure out a stride for the current entity that makes it so this entity
# gets roughly as many snippets made as the entity with most text in the db
def get_stride(texts_length, entity_max_snippets):
    # min and max force output to be in the rane 1..settings max stride
    stride = int(max(texts_length / entity_max_snippets, 1))
    return min(stride, SETTINGS['stride'])

# there can be multiple texts saved in the db for each person
# this function snippetizes each text in turn and returns a list of snippets
# from all the texts
def process_texts(texts, entity_max_snippets, settings):
    snippet_list = []
    texts_length = get_texts_length(texts)
    stride = get_stride(texts_length, entity_max_snippets)

    text = concat_all_texts(texts)

    text = remove_punctuation(text)
    if len(text.split()) > 0:
        snippets = process_text(text, settings, stride)
        # so all entities will have same number of snippets, unless too massive
        while len(snippets) < entity_max_snippets:
            snippets = snippets * 2
        snippets = snippets[:entity_max_snippets]
        for snippet in snippets:
            snippet_list.append(snippet)
    return snippet_list


def get_entity_snippets(entity, entity_max_snippets, settings):
    return process_texts(entity['texts'], entity_max_snippets, settings)


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
    longest_texts_length = get_longest_texts(handler)
    entity_max_snippets = get_max_snippets(longest_texts_length, SETTINGS['stride'])
    for entity in handler.get_entity_iterator():
        # if to make sure there actually are some texts for this entity
        if len(entity['texts']) > 0:
            count += 1
            snippets = get_entity_snippets(entity, entity_max_snippets, SETTINGS)
            print("writing snippets for " + str(entity['_id']) + " number " + str(count) + " of " + str(num_entities))
            write_snippets(handler, entity, snippets)

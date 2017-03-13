import pickle
import embeddings

fp = open('./pickled_data/buckets_map', 'r')
buckets = pickle.load(fp)
fp.close()
current_indices = {'training':{}, 'test':{}, 'validation':{}}

def reset_indices(data_type):
    buckets = current_indices[data_type].keys()
    for bucket in buckets:
        current_indices[data_type][bucket] = 0

def has_more(bucket, data_type):
    index = current_indices[data_type][bucket]
    return index < len(buckets[data_type][bucket])

def get_batch(bucket, batch_size, data_type):
    if bucket not in buckets[data_type]:
        return []
    if bucket not in current_indices[data_type]:
        current_indices[data_type][bucket] = 0
    batch = []
    for i in xrange(batch_size):
        index = current_indices[data_type][bucket]
        files_list = buckets[data_type][bucket]
        if index >= len(files_list):
            break
        sentences, labels = process_document(files_list[index])
        current_indices[data_type][bucket] += 1
        if len(sentences) == 0:
            continue    # the signal that document has at least one sentence with more than 60 words
        batch.append((sentences, labels))
    return batch

# Returns the input and label for each sentence in the document
def process_document(path):
    path = '/home/putama/Documents/neuralsum/'+path[2:]
    doc_fp = open(path, 'r')
    article = doc_fp.read().split('\n\n')[1]
    entity_map = get_entity_map(doc_fp)
    doc_fp.close()
    sentences = []
    labels = []
    for sentence_string in article.split('\n'):
        split_sentence = sentence_string.split()
        label = 1 if int(split_sentence.pop()) == 1 else 0
        split_sentence.insert(0, '<<GO>>')
        split_sentence.append('<<EOS>>')
        if len(split_sentence) > 60:
            return [[],[]]   # Signal that the document contains a sentence of length greater than 60
        for i,word in enumerate(split_sentence):
            if word[0] == '@' and word in entity_map:
                split_sentence[i] = entity_map[word]
        while len(split_sentence) < 60:
            split_sentence.append('<<SENTENCE_PAD>>')
        sentences.append(split_sentence)
        labels.append(label)
    pad_doc(sentences, labels)
    set_to_indexes(sentences)
    return sentences, labels

# Adds padding sentences to get the number of sentences to the bucket size,
# and adds a starting <<GO>> sentence.
def pad_doc(sentences, labels):
    while len(sentences)%10 != 0:
        sentences.append(['<<DOC_PAD>>' for x in xrange(60)])
        labels.append(0)
    sentences.insert(0, ['<<GO>>' for x in xrange(60)])

# Converts a list of lists of words to a list of lists of word indexes
def set_to_indexes(sentences):
    for sentence in sentences:
        for i in xrange(len(sentence)):
            word = sentence[i]
            index = embeddings.get_index(word) if embeddings.embedding_exists(word) else embeddings.get_index('<<UNK>>')
            sentence[i] = index

def get_entity_map(doc_fp):
    doc_fp.seek(0)
    pieces = doc_fp.read().split('\n\n')
    if len(pieces) < 4:
        return {}
    entity_string = pieces[3]
    entity_map = {}
    for line in entity_string.split('\n'):
        split_line = line.split(':')
        if len(split_line) < 2:
            continue
        entity_map[line.split(':')[0]] = line.split(':')[1]
    return entity_map

a = get_batch(20, 5, 'test')
print len(a)
while has_more(20, 'test'):
    a = get_batch(20, 5, 'test')
    print len(a)
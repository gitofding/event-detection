import re
import numpy as np

class Text2Digit():
    def __init__(self, config):
        self.train_vals = load_from_file(config.train)
        self.dev_vals = load_from_file(config.dev)
        self.test_vals = load_from_file(config.test)
        self.token_loc = 0
        self.atag_loc = 1

        self.tokens = self.train_vals[self.token_loc]+self.dev_vals[self.token_loc]+self.test_vals[self.token_loc]
        self.atags = self.train_vals[self.atag_loc]+self.dev_vals[self.atag_loc]+self.test_vals[self.atag_loc]
        self.atag_dict = prep_tag_dict(self.atags)

        target_tokens = list(set([w for words in self.tokens for w in words]))
        pretrain_embedding, pretrain_vocab = loadPretrain(config.pretrain_embed, target_words=None)
        self.unk_id = pretrain_vocab['<unk>']
        self.vocab = pretrain_vocab
        self.pretrain_embedding = pretrain_embedding

        self.get_digit_data()

    def get_digit_data(self):
        digit_train_tokens = digitalize(self.train_vals[self.token_loc], self.vocab, self.unk_id)
        digit_train_atags = digitalize(self.train_vals[self.atag_loc], self.atag_dict, None)

        digit_dev_tokens = digitalize(self.dev_vals[self.token_loc], self.vocab, self.unk_id)
        digit_dev_atags = digitalize(self.dev_vals[self.atag_loc], self.atag_dict, None)

        digit_test_tokens = digitalize(self.test_vals[self.token_loc], self.vocab, self.unk_id)
        digit_test_atags = digitalize(self.test_vals[self.atag_loc], self.atag_dict, None)

        self.train = (digit_train_tokens, digit_train_atags)
        self.dev = (digit_dev_tokens, digit_dev_atags)
        self.test = (digit_test_tokens, digit_test_atags)

def prep_tag_dict(tags_in_sents):
    tags_data = sorted(list(set([tag for tags in tags_in_sents for tag in tags if tag != "O"])))
    tags_data = dict(zip(tags_data, range(1, len(tags_data)+1)))
    tags_data["O"] = 0
    return tags_data

# arr: list
def add_st_ed(arr, st_token, ed_token):
    arr.insert(0, st_token)
    arr.append(ed_token)
    return arr

def load_from_file(filename):
    '''
        Input:
            one line one sentence format. eg: w1 w2 w3[\t]at1 at2 at3[\t]bt1 bt2 bt3[\t]ct11, ct12, ct13[\t]ct21, ct22, ct23[\t]ct31, ct32, ct33
            atag: event trigger tags
            btag: entity tags
            ctag: argument roles
        Output:
            tokens: [w1, w2, w3]
            atags:  [at1, at2, at3]
     '''
    content = open(filename, "r").readlines()
    content = [line.rstrip("\n").split("\t") for line in content if len(line.strip())>1]
    content = content[:5000]

    data = [[item.strip().split() for item in line] for line in content]
    tokens = [item[0] for item in data]
    atags  = [item[1] for item in data]
    tokens = process_tokens(tokens)
    return_values = []
    return_values.append(tokens)
    return_values.append(atags)
    return return_values

def loadPretrain(filepath, target_words=None):
    unk_word = "<unk>"
    st = '<s>'
    ed = '</s>'
    content = open(filepath, "r").readlines()

    if target_words is not None and unk_word not in target_words: target_words.append(unk_word)
    pretrain_embedding = []
    pretrain_vocab = {} # word: word_id
    for word_id, line in enumerate(content):
        word_item = line.strip().split()
        if len(word_item) == 2: continue
        word_text = word_item[0]
        if target_words is not None and word_text not in target_words: continue
        if(len(word_item)!=301): print(len(word_item), word_item)
        embed_word = [float(item) for item in word_item[1:]]
        pretrain_embedding.append(embed_word)
        pretrain_vocab[word_text] = len(pretrain_vocab)

    # add unk_word
    word_dim = len(pretrain_embedding[-1])
    if st not in pretrain_vocab:
        pretrain_vocab[st] = len(pretrain_vocab)
        pretrain_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())
    if ed not in pretrain_vocab:
        pretrain_vocab[ed] = len(pretrain_vocab)
        pretrain_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())
    if unk_word not in pretrain_vocab:
        pretrain_vocab[unk_word] = len(pretrain_vocab)
        pretrain_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())
    return np.matrix(pretrain_embedding), pretrain_vocab

def words2ids(arr, vocab, unk_id):
    new_arr = [vocab[witem] if witem in vocab else unk_id for witem in arr]
    return new_arr

def digitalize(arr, vocab, unk_id, sepchar=None):
    return [words2ids(item, vocab, unk_id) for item in arr]

def process_word(token):
    word = ""
    if len(token)==4 and token[:2] in ["17", "18", "19", "20"]: # year
        word = token
    else:
        for c in token:
            if c.isdigit(): word += '0'
            else: word += c
        word = re.sub("0+", "0", word)
    return word.lower()

def process_tokens(all_tokens):
    return [[process_word(w) for w in sent] for sent in all_tokens]


'''
sources from https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/execution?select=embeddings.zip
'''

import re
import os
import io
import pandas as pd
import numpy as np
import operator

root_dir = '/home/ace19/dl_data/shopee-product-matching'

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                       "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                       "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                       "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                       "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                       "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                       "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                       "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
                       "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have", "you're": "you are", "you've": "you have"}

# Remove any special characters
# https://www.utf8-chartable.de/unicode-utf8-table.pl?start=9984&number=128&names=-&utf8=string-literal
special_characters_emojis = ['\xe2\x9c\x80', '\xe2\x9c\x81', '\xe2\x9c\x82', '\xe2\x9c\x83', '\xe2\x9c\x84',
                             '\xe2\x9c\x85',
                             '\xe2\x9c\x86', '\xe2\x9c\x87', '\xe2\x9c\x88', '\xe2\x9c\x89', '\xe2\x9c\x8a',
                             '\xe2\x9c\x8b',
                             '\xe2\x9c\x8c', '\xe2\x9c\x8d', '\xe2\x9c\x8e', '\xe2\x9c\x8f', '\xe2\x9c\x90',
                             '\xe2\x9c\x91',
                             '\xe2\x9c\x92', '\xe2\x9c\x93', '\xe2\x9c\x94', '\xe2\x9c\x95', '\xe2\x9c\x96',
                             '\xe2\x9c\x97',
                             '\xe2\x9c\x98', '\xe2\x9c\x99', '\xe2\x9c\x9a', '\xe2\x9c\x9b', '\xe2\x9c\x9c',
                             '\xe2\x9c\x9d',
                             '\xe2\x9c\x9e', '\xe2\x9c\x9f', '\xe2\x9c\xa0', '\xe2\x9c\xa1', '\xe2\x9c\xa2',
                             '\xe2\x9c\xa3',
                             '\xe2\x9c\xa4', '\xe2\x9c\xa5', '\xe2\x9c\xa6', '\xe2\x9c\xa7', '\xe2\x9c\xa8',
                             '\xe2\x9c\xa9',
                             '\xe2\x9c\xaa', '\xe2\x9c\xab', '\xe2\x9c\xac', '\xe2\x9c\xad', '\xe2\x9c\xae',
                             '\xe2\x9c\xaf',
                             '\xe2\x9c\xb0', '\xe2\x9c\xb1', '\xe2\x9c\xb2', '\xe2\x9c\xb3', '\xe2\x9c\xb4',
                             '\xe2\x9c\xb5',
                             '\xe2\x9c\xb6', '\xe2\x9c\xb7', '\xe2\x9c\xb8', '\xe2\x9c\xb9', '\xe2\x9c\xba',
                             '\xe2\x9c\xbb',
                             '\xe2\x9c\xbc', '\xe2\x9c\xbd', '\xe2\x9c\xbe', '\xe2\x9c\xbf', '\xe2\x9d\x80',
                             '\xe2\x9d\x81',
                             '\xe2\x9d\x82', '\xe2\x9d\x83', '\xe2\x9d\x84', '\xe2\x9d\x85', '\xe2\x9d\x86',
                             '\xe2\x9d\x87',
                             '\xe2\x9d\x88', '\xe2\x9d\x89', '\xe2\x9d\x8a', '\xe2\x9d\x8b', '\xe2\x9d\x8c',
                             '\xe2\x9d\x8d',
                             '\xe2\x9d\x8e', '\xe2\x9d\x8f', '\xe2\x9d\x90', '\xe2\x9d\x91', '\xe2\x9d\x92',
                             '\xe2\x9d\x93',
                             '\xe2\x9d\x94', '\xe2\x9d\x95', '\xe2\x9d\x96', '\xe2\x9d\x97', '\xe2\x9d\x98',
                             '\xe2\x9d\x99',
                             '\xe2\x9d\x9a', '\xe2\x9d\x9b', '\xe2\x9d\x9c', '\xe2\x9d\x9d', '\xe2\x9d\x9e',
                             '\xe2\x9d\x9f',
                             '\xe2\x9d\xa0', '\xe2\x9d\xa1', '\xe2\x9d\xa2', '\xe2\x9d\xa3', '\xe2\x9d\xa4',
                             '\xe2\x9d\xa5',
                             '\xe2\x9d\xa6', '\xe2\x9d\xa7', '\xe2\x9d\xa8', '\xe2\x9d\xa9', '\xe2\x9d\xaa',
                             '\xe2\x9d\xab',
                             '\xe2\x9d\xac', '\xe2\x9d\xad', '\xe2\x9d\xae', '\xe2\x9d\xaf', '\xe2\x9d\xb0',
                             '\xe2\x9d\xb1',
                             '\xe2\x9d\xb2', '\xe2\x9d\xb3', '\xe2\x9d\xb4', '\xe2\x9d\xb5', '\xe2\x9d\xb6',
                             '\xe2\x9d\xb7',
                             '\xe2\x9d\xb8', '\xe2\x9d\xb9', '\xe2\x9d\xba', '\xe2\x9d\xbb', '\xe2\x9d\xbc',
                             '\xe2\x9d\xbd',
                             '\xe2\x9d\xbe', '\xe2\x9d\xbf', ]

# https://www.utf8-chartable.de/unicode-utf8-table.pl?names=-&utf8=string-literal
special_characters_mapping = {'\\xc2\\xa1': '¡', '\\xc2\\xa2': '¢', '\\xc2\\xa3': '£', '\\xc2\\xa4': '¤',
                              '\\xc2\\xa5': '¥', '\\xc2\\xa6': '¦', '\\xc2\\xa7': '§', '\\xc2\\xa8': '¨',
                              '\\xc2\\xa9': '©', '\\xc2\\xaa': 'ª', '\\xc2\\xab': '«', '\\xc2\\xac': '¬',
                              '\\xc2\\xad': '­', '\\xc2\\xae': '®', '\\xc2\\xaf': '¯', '\\xc2\\xb0': '°',
                              '\\xc2\\xb1': '±', '\\xc2\\xb2': '²', '\\xc2\\xb3': '³', '\\xc2\\xb4': '´',
                              '\\xc2\\xb5': 'µ', '\\xc2\\xb6': '¶', '\\xc2\\xb7': '·', '\\xc2\\xb8': '¸',
                              '\\xc2\\xb9': '¹', '\\xc2\\xba': 'º', '\\xc2\\xbb': '»', '\\xc2\\xbc': '¼',
                              '\\xc2\\xbd': '½', '\\xc2\\xbe': '¾', '\\xc2\\xbf': '¿', '\\xc3\\x80': 'À',
                              '\\xc3\\x81': 'Á', '\\xc3\\x82': 'Â', '\\xc3\\x83': 'Ã', '\\xc3\\x84': 'Ä',
                              '\\xc3\\x85': 'Å', '\\xc3\\x86': 'Æ', '\\xc3\\x87': 'Ç', '\\xc3\\x88': 'È',
                              '\\xc3\\x89': 'É', '\\xc3\\x8a': 'Ê', '\\xc3\\x8b': 'Ë', '\\xc3\\x8c': 'Ì',
                              '\\xc3\\x8d': 'Í', '\\xc3\\x8e': 'Î', '\\xc3\\x8f': 'Ï', '\\xc3\\x90': 'Ð',
                              '\\xc3\\x91': 'Ñ', '\\xc3\\x92': 'Ò', '\\xc3\\x93': 'Ó', '\\xc3\\x94': 'Ô',
                              '\\xc3\\x95': 'Õ', '\\xc3\\x96': 'Ö', '\\xc3\\x97': '×', '\\xc3\\x98': 'Ø',
                              '\\xc3\\x99': 'Ù', '\\xc3\\x9a': 'Ú', '\\xc3\\x9b': 'Û', '\\xc3\\x9c': 'Ü',
                              '\\xc3\\x9d': 'Ý', '\\xc3\\x9e': 'Þ', '\\xc3\\x9f': 'ß', '\\xc3\\xa0': 'à',
                              '\\xc3\\xa1': 'á', '\\xc3\\xa2': 'â', '\\xc3\\xa3': 'ã', '\\xc3\\xa4': 'ä',
                              '\\xc3\\xa5': 'å', '\\xc3\\xa6': 'æ', '\\xc3\\xa7': 'ç', '\\xc3\\xa8': 'è',
                              '\\xc3\\xa9': 'é', '\\xc3\\xaa': 'ê', '\\xc3\\xab': 'ë', '\\xc3\\xac': 'ì',
                              '\\xc3\\xad': 'í', '\\xc3\\xae': 'î', '\\xc3\\xaf': 'ï', '\\xc3\\xb0': 'ð',
                              '\\xc3\\xb1': 'ñ', '\\xc3\\xb2': 'ò', '\\xc3\\xb3': 'ó', '\\xc3\\xb4': 'ô',
                              '\\xc3\\xb5': 'õ', '\\xc3\\xb6': 'ö', '\\xc3\\xb7': '÷', '\\xc3\\xb8': 'ø',
                              '\\xc3\\xb9': 'ù', '\\xc3\\xba': 'ú', '\\xc3\\xbb': 'û', '\\xc3\\xbc': 'ü',
                              '\\xc3\\xbd': 'ý', '\\xc3\\xbe': 'þ', '\\xc3\\xbf': 'ÿ', }

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


# https://fasttext.cc/docs/en/crawl-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# Vocabulary functions
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# Coverage functions
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/execution?select=embeddings.zip
def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='utf-8'))

    return embeddings_index


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def clean_special_chars2(text, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    return text


# ---------------------------------------------------------------

import emoji


# https://stackoverflow.com/questions/51784964/remove-emojis-from-multilingual-unicode-text
# def remove_emoji(text):
#     return emoji.get_emoji_regexp().sub(u'', text)
def clean_emojjs(text):
    # # https://stackoverflow.com/questions/51217909/removing-all-emojis-from-text
    # emoji_pattern = re.compile("["
    #                            u"\U0001F600-\U0001F64F"  # emoticons
    #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #                            u"\U0001F1F2-\U0001F1F4"  # Macau flag
    #                            u"\U0001F1E6-\U0001F1FF"  # flags
    #                            u"\U0001F600-\U0001F64F"
    #                            u"\U00002702-\U000027B0"
    #                            u"\U000024C2-\U0001F251"
    #                            u"\U0001f926-\U0001f937"
    #                            u"\U0001F1F2"
    #                            u"\U0001F1F4"
    #                            u"\U0001F620"
    #                            u"\u200d"
    #                            u"\u2640-\u2642"
    #                            u"U+2700-U+277F"
    #                            "]+", flags=re.UNICODE)
    #
    # clean_text = emoji_pattern.sub(r'', text)
    text = re.sub(r'(\\...)*', '', text)

    return text


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


train_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))

# Loading embeddings
# print("Extracting English embedding")
# embed_en = load_embed(os.path.join(root_dir, 'cc.en.300.vec'))
print("Extracting Indonesian embedding")
embed_id = load_embed(os.path.join(root_dir, 'cc.id.300.vec'))

# starting point
vocab = build_vocab(train_df['title'])
# print("english : ")
# oov_en = check_coverage(vocab, embed_en)
print("indonesian : ")
oov_id = check_coverage(vocab, embed_id)

# 1. let us lower our texts :
print('\n**************************')
print('lower texts')
print('**************************')
train_df['lowered_title'] = train_df['title'].apply(lambda x: x.lower())
vocab_low = build_vocab(train_df['lowered_title'])
# print("english : ")
# oov_en = check_coverage(vocab_low, embed_en)
print("indonesian : ")
oov_id = check_coverage(vocab_low, embed_id)


print('\n**************************')
print('add words & lower texts')
print('**************************')
# print("english : ")
# add_lower(embed_en, vocab)
print("indonesian : ")
add_lower(embed_id, vocab)

# print("english : ")
# oov_en = check_coverage(vocab_low, embed_en)
print("indonesian : ")
oov_id = check_coverage(vocab_low, embed_id)

# print('\noov_en[:20]', oov_en[:20])
print('oov_id[:20]', oov_id[:20])



# let us deal with special characters
print('\n**************************')
print('let us deal with special characters')
print('**************************')

# print("english :")
# print(unknown_punct(embed_en, punct))
print("indonesian :")
print(unknown_punct(embed_id, punct))

# train_df['treated_title'] = \
#     train_df['lowered_title'].apply(lambda x: clean_special_chars2(x, special_characters_mapping))
# vocab2 = build_vocab(train_df['treated_title'])
# # print("english : ")
# # oov_en = check_coverage(vocab, embed_en)
# print("indonesian : ")
# oov_id = check_coverage(vocab2, embed_id)

train_df['treated_title'] = \
    train_df['lowered_title'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
vocab = build_vocab(train_df['treated_title'])
# print("english : ")
# oov_en = check_coverage(vocab, embed_en)
print("indonesian : ")
oov_id = check_coverage(vocab, embed_id)


# 0. remove emojjs
print('\n**************************')
print('remove emojjs')
print('**************************')
train_df['no_emojjs_title'] = train_df['lowered_title'].apply(lambda x: clean_emojjs(x))
vocab_no_emojjs = build_vocab(train_df['no_emojjs_title'])
# print("english : ")
# oov_en = check_coverage(vocab_no_emojjs, embed_en)
print("indonesian : ")
oov_id = check_coverage(vocab_no_emojjs, embed_id)


# 3. FastText does not understand contractions
print('\n**************************')
print('We use the map to replace contractions')
print('**************************')

train_df['treated_title'] = \
    train_df['no_emojjs_title'].apply(lambda x: clean_contractions(x, contraction_mapping))

vocab = build_vocab(train_df['treated_title'])
# print("english : ")
# oov_en = check_coverage(vocab, embed_en)
print("indonesian : ")
oov_id = check_coverage(vocab, embed_id)


# print('\noov_en[:50]', oov_en[:50])
print('oov_id[:50]', oov_id[:50])

# TODO: correct manually most frequent mispells
# mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
#                 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
#                 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
#                 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
#                 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many',
#                 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
#                 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',
#                 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
#                 '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
#                 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
#                 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
#
# train_df['treated_question'] = train_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
#
# vocab = build_vocab(train_df['treated_question'])
# print("english : ")
# oov_en = check_coverage(vocab, embed_en)
# print("indonesian : ")
# oov_id = check_coverage(vocab, embed_id)

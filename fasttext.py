import os
import io
import re
import timeit
import string
import pandas as pd
import numpy as np
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from gensim import utils
from gensim.utils import tokenize
from gensim.test.utils import datapath

root_dir = '/home/ace19/dl_data/shopee-product-matching'

default_stop_words = ['atau', 'dan', 'and', 'murah', 'grosir',
                      'untuk', 'termurah', 'cod', 'terlaris', 'bisacod', 'terpopuler',
                      'bisa', 'terbaru', 'tempat', 'populer', 'di', 'sale', 'bayar', 'flash',
                      'promo', 'seler', 'in', 'salee', 'diskon', 'gila', 'starseller', 'seller']

# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/execution?select=embeddings.zip
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }

# TODO: fix
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
# punct = "/'?!,#$%\'()*+/:;<=>@[\\]^`{|}" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# https://www.utf8-chartable.de/unicode-utf8-table.pl?names=-&utf8=string-literal
special_characters_mapping = {'\\xc2\\xa1': '¡', '\\xc2\\xa2': '¢', '\\xc2\\xa3': 'e', '\\xc2\\xa4': '¤',
                              '\\xc2\\xa5': '¥', '\\xc2\\xa6': '¦', '\\xc2\\xa7': '§', '\\xc2\\xa8': '¨',
                              '\\xc2\\xa9': '©', '\\xc2\\xaa': 'ª', '\\xc2\\xab': '«', '\\xc2\\xac': '¬',
                              '\\xc2\\xad': '­', '\\xc2\\xae': '®', '\\xc2\\xaf': '¯', '\\xc2\\xb0': '°',
                              '\\xc2\\xb1': '±', '\\xc2\\xb2': '²', '\\xc2\\xb3': '³', '\\xc2\\xb4': '´',
                              '\\xc2\\xb5': 'µ', '\\xc2\\xb6': '¶', '\\xc2\\xb7': '·', '\\xc2\\xb8': '¸',
                              '\\xc2\\xb9': '¹', '\\xc2\\xba': 'º', '\\xc2\\xbb': '»', '\\xc2\\xbc': '¼',
                              '\\xc2\\xbd': '½', '\\xc2\\xbe': '¾', '\\xc2\\xbf': '¿', '\\xc3\\x80': 'A',
                              '\\xc3\\x81': 'A', '\\xc3\\x82': 'A', '\\xc3\\x83': 'A', '\\xc3\\x84': 'A',
                              '\\xc3\\x85': 'A', '\\xc3\\x86': 'Æ', '\\xc3\\x87': 'Ç', '\\xc3\\x88': 'E',
                              '\\xc3\\x89': 'E', '\\xc3\\x8a': 'E', '\\xc3\\x8b': 'E', '\\xc3\\x8c': 'I',
                              '\\xc3\\x8d': 'I', '\\xc3\\x8e': 'I', '\\xc3\\x8f': 'I', '\\xc3\\x90': 'Ð',
                              '\\xc3\\x91': 'N', '\\xc3\\x92': 'Ò', '\\xc3\\x93': 'Ó', '\\xc3\\x94': 'Ô',
                              '\\xc3\\x95': 'Õ', '\\xc3\\x96': 'Ö', '\\xc3\\x97': '×', '\\xc3\\x98': 'Ø',
                              '\\xc3\\x99': 'U', '\\xc3\\x9a': 'U', '\\xc3\\x9b': 'U', '\\xc3\\x9c': 'U',
                              '\\xc3\\x9d': 'Y', '\\xc3\\x9e': 'Þ', '\\xc3\\x9f': 'ß', '\\xc3\\xa0': 'a',
                              '\\xc3\\xa1': 'a', '\\xc3\\xa2': 'a', '\\xc3\\xa3': 'a', '\\xc3\\xa4': 'a',
                              '\\xc3\\xa5': 'a', '\\xc3\\xa6': 'æ', '\\xc3\\xa7': 'ç', '\\xc3\\xa8': 'e',
                              '\\xc3\\xa9': 'e', '\\xc3\\xaa': 'e', '\\xc3\\xab': 'e', '\\xc3\\xac': 'i',
                              '\\xc3\\xad': 'i', '\\xc3\\xae': 'i', '\\xc3\\xaf': 'i', '\\xc3\\xb0': 'ð',
                              '\\xc3\\xb1': 'n', '\\xc3\\xb2': 'o', '\\xc3\\xb3': 'o', '\\xc3\\xb4': 'o',
                              '\\xc3\\xb5': 'o', '\\xc3\\xb6': 'o', '\\xc3\\xb7': '÷', '\\xc3\\xb8': 'ø',
                              '\\xc3\\xb9': 'u', '\\xc3\\xba': 'u', '\\xc3\\xbb': 'u', '\\xc3\\xbc': 'u',
                              '\\xc3\\xbd': 'y', '\\xc3\\xbe': 'þ', '\\xc3\\xbf': 'y', }


def preprocess_text(text):
    s = str(text).lower()
    # replace & with and
    #     s = re.sub('&', ' and ', s)
    #     # replace / with or (idn)
    #     s = re.sub('/', 'atau', s, count=1)
    # remove emojj
    s = re.sub(r'(\\...)*', '', s)
    #
    #     s = clean_special_chars(s)
    #     # remove all special characters
    #     s = re.sub(r"[^a-zA-Z0-9]+", ' ', s)
    #     # replace 's with only s (the special character ' is not the standard one, hence the implementation)
    #     s = re.sub(' s ', 's ', s)
    # # add whitespace after each number
    # s = re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", s).strip()
    return s


def clean_text(title):
    words = word_tokenize(title)
    clean_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words('indonesian') and word not in default_stop_words:  # 불용어 제거 e.g.
            stemmer = SnowballStemmer('english')
            word = stemmer.stem(word)  # 어간 추출
            clean_words.append(word)
    #             try:
    #                 float(word)
    #             except:
    #                 clean_words.append(word)

    # print(clean_words)
    return ' '.join(clean_words)


def removePunctuation(text):
    punc_translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(punc_translator)


def clean_special_chars(text):
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])

    for p in special_characters_mapping:
        text = text.replace(p, special_characters_mapping[p])

    # remove emojj
    text = re.sub(r'(\\...)*', '', text)

    #     for p in punct:
    #         text = text.replace(p, f' {p} ')

    #     specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': ''}  # Other special characters that I have to deal with in last
    #     for s in specials:
    #         text = text.replace(s, specials[s])

    return text


# #######################
# 1. make corpus file
# #######################
# df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
#
# df['preprocess_title'] = df['title'].apply(lambda x: preprocess_text(x))
# df['title_clean'] = df['preprocess_title'].apply(lambda x: removePunctuation(x))
#
# f = open("results/corpus.txt", 'w')
# for title in tqdm(df['title_clean'].values.tolist()):
#     f.write(title + '\n')
# f.close()

corpus_file = 'results/corpus.txt'
# new_sent = []
# f = open(corpus_file, 'r')
# lines = f.readlines()
# for line in lines:
#     title = []
#     for t in line.strip().split(" "):
#         try:
#             float(t)
#         except:
#             title.append(t)
#     new_sent.append(title)
# f.close()


# ############################
# train from scratch
# ############################
shopee_model = FastText(vector_size=768, seed=8)
shopee_model.build_vocab(corpus_file=corpus_file)  # scan over corpus to build the vocabulary
total_words = shopee_model.corpus_total_words  # number of words in the corpus
print('total_words: ', total_words)
# shopee_model.train(corpus_file=corpus_file, total_words=total_words, epochs=50)
# print('shopee_model.wv.vectors.shape', shopee_model.wv.vectors.shape)

# ############################
# read pre-trained model.
# ############################
# read pre-trained model.
fb_model = load_facebook_model(os.path.join(root_dir, 'word_vectors', 'cc.id.300.bin'))
print('fb_model.wv.vectors.shape', fb_model.wv.vectors.shape)
#
#
# #################################
# 3. train from pre-trained model
# #################################
# new_sent = [['lord', 'of', 'the', 'rings'], ['lord', 'of', 'the', 'flies']]
fb_model.build_vocab(corpus_file=corpus_file, update=True)
# total_words = fb_model.corpus_total_words  # number of words in the corpus
# print('total_words: ', total_words)
fb_model.train(corpus_file=corpus_file, total_words=total_words, epochs=50)
print('fb_model.wv.vectors.shape', fb_model.wv.vectors.shape)
# >> (2000000, 300)


# ################################
# 4. save & load
# ################################
# fname = 'results/shopee_fasttext.model'
# shopee_model.save(fname)

fname = 'results/fb_shopee_fasttext.model'
fb_model.save(fname)

# fb_model = FastText.load(fname)


# ################################
# *. test sentence embeddings
# ################################
# vect = fb_model.wv['telepon']
# vect2 = fb_model.wv['Lemonilo']
# vect3 = fb_model.wv['lampu']
# sent_vect = [vect, vect2, vect3]
# vects = np.stack(sent_vect)
# result = np.mean(vects, axis=0)
#
# print('')

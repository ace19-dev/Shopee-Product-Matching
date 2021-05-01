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

added_stopwords = ['grosir', 'cod', 'diskon', 'starseller']

# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/execution?select=embeddings.zip
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }

# punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct = "/'?!,#$%\'()*+/:;<=>@[\\]^`{|}" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

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


def clean_text(title):
    words = word_tokenize(title)
    clean_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words('indonesian') and word not in added_stopwords:  # 불용어 제거 e.g.
            stemmer = SnowballStemmer('english')
            word = stemmer.stem(word)  # 어간 추출
            clean_words.append(word)
    # print(clean_words)
    return ' '.join(clean_words)


def removePunctuation(text):
    punc_translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(punc_translator)


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


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


# TODO: fix.
def clean_special_chars(text):
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])

    for p in special_characters_mapping:
        text = text.replace(p, special_characters_mapping[p])

    text = re.sub(r'(\\...)*', '', text)

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text


# TODO: fix
# https://stackoverflow.com/questions/51217909/removing-all-emojis-from-text
# def clean_emojjs(text):
# text2 = repr(text)
# text3 = codecs.decode(text2, 'unicode_escape')
# # TODO: fix
# emoji_pattern = re.compile("["
#                            u"\U0001F600-\U0001F64F"  # emoticons
#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            u"\U00002500-\U00002BEF"  # chinese char
#                            u"\U00002702-\U000027B0"
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
#                            u"\U0001f926-\U0001f937"
#                            u"\U00010000-\U0010ffff"
#                            u"\u2640-\u2642"
#                            u"\u2600-\u2B55"
#                            u"\u200d"
#                            u"\u23cf"
#                            u"\u23e9"
#                            u"\u231a"
#                            u"\ufe0f"  # dingbats
#                            u"\u3030"
#                            u"\u2680-\u277F"
#                            "]+", flags=re.UNICODE)
# return emoji_pattern.sub(r'', text)


def clean_emojjs(text):
    # text = re.sub(r'(\\...)*', '', text)

    for p in punct:
        text = text.replace(p, '')

    return text


# #######################
# 1. make corpus file
# #######################
# df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
#
# df['lowered_title'] = df['title'].apply(lambda x: x.lower())
# df['clean_special_chars_title'] = df['lowered_title'].apply(lambda x: clean_special_chars(x))
# df['clean_emojjs_title'] = df['clean_special_chars_title'].apply(lambda x: clean_emojjs(x))
# df['title_clean'] = df['clean_emojjs_title'].apply(lambda x: clean_text(x))
#
# f = open("corpus.txt", 'w')
# for title in tqdm(df['title_clean'].values.tolist()):
#     f.write(title+'\n')
# f.close()


corpus_file = 'corpus.txt'
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
scratch_model = FastText(vector_size=512, window=3, min_count=1)
scratch_model.build_vocab(corpus_file=corpus_file)  # scan over corpus to build the vocabulary
total_words = scratch_model.corpus_total_words  # number of words in the corpus
print('total_words: ', total_words)
scratch_model.build_vocab(corpus_file=corpus_file)
scratch_model.train(corpus_file=corpus_file, total_words=total_words, epochs=100)
print('scratch_model.wv.vectors.shape', scratch_model.wv.vectors.shape)


# # ############################
# # read pre-trained model.
# # ############################
# # read pre-trained model.
# fb_model = load_facebook_model(os.path.join(root_dir, 'word_vectors', 'cc.id.300.bin'))
# print('fb_model.wv.vectors.shape', fb_model.wv.vectors.shape)
#
#
# # #################################
# # 3. train from pre-trained model
# # #################################
# # new_sent = [['lord', 'of', 'the', 'rings'], ['lord', 'of', 'the', 'flies']]
# fb_model.build_vocab(corpus_file=corpus_file, update=True)
# # total_words = fb_model.corpus_total_words  # number of words in the corpus
# # print('total_words: ', total_words)
# fb_model.train(corpus_file=corpus_file, total_words=total_words, epochs=100)
# print('fb_model.wv.vectors.shape', fb_model.wv.vectors.shape)


# ################################
# 4. save & load
# ################################
fname = 'results/scratch_fasttext.model'
scratch_model.save(fname)
# fb_model.save(fname)
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

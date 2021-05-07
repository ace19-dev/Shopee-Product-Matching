import re
import os
import string
# import codecs
# import emoji
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# print(stopwords.fileids())

NUM_CLASS = 11014
MAX_LENGTH = 32

# factory = StemmerFactory()

default_stop_words = ['atau', 'dan', 'and', 'murah', 'grosir',
                      'untuk', 'termurah', 'cod', 'terlaris', 'bisacod', 'terpopuler',
                      'bisa', 'terbaru', 'tempat', 'populer', 'di', 'sale', 'bayar', 'flash',
                      'promo', 'seler', 'in', 'salee', 'diskon', 'gila', 'starseller', 'seller']

# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/execution?select=embeddings.zip
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

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
            #             stemmer = SnowballStemmer('english')
            #             word = stemmer.stem(word)  # 어간 추출
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


# https://stackoverflow.com/questions/51217909/removing-all-emojis-from-text
# def clean_emojjs(text):
#     emoji_pattern = re.compile("["
#                                u"\U0001F600-\U0001F64F"  # emoticons
#                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                                u"\U0001F1F2-\U0001F1F4"  # Macau flag
#                                u"\U0001F1E6-\U0001F1FF"  # flags
#                                u"\U0001F600-\U0001F64F"
#                                u"\U00002702-\U000027B0"
#                                u"\U000024C2-\U0001F251"
#                                u"\U0001f926-\U0001f937"
#                                u"\U0001F1F2"
#                                u"\U0001F1F4"
#                                u"\U0001F620"
#                                u"\u200d"
#                                u"\u2640-\u2642"
#                                u"U+2700-U+277F"
#                                "]+", flags=re.UNICODE)

#     return emoji_pattern.sub(r'', text)


def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort=pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    # df = df.reset_index()
    df = df.drop('sort', axis=1)
    return df


class ProductTextDataset(data.Dataset):
    def __init__(self, data_dir, fold, csv, mode, tokenizer):
        self.data_dir = data_dir
        self.fold = fold
        self.csv = csv
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH
        self.num_classes = NUM_CLASS

        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f, encoding='utf_8') for f in self.csv])
        self.preprocess()

        samples = list(np.concatenate([np.load(data_dir + '/fold/%s' % f, allow_pickle=True) for f in self.fold]))
        self.df = df_loc_by_list(self.df, 'posting_id', samples)
        self.labels = self.df['label'].values

        self.df['preprocess_title'] = self.df['title'].apply(lambda x: preprocess_text(x))
        self.df['title_clean'] = self.df['preprocess_title'].apply(lambda x: removePunctuation(x))
        self.encodings = tokenizer(self.df['title_clean'].values.tolist(),
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length)

    def __getitem__(self, index):
        # putting each tensor in front of the corresponding key from the tokenizer
        # HuggingFace tokenizers give you whatever you need to feed to the corresponding model
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        # when testing, there are no targets so we won't do the following
        item['labels'] = torch.tensor(self.labels[index]).long()

        return item

    def __str__(self):
        length = len(self)

        string = ''
        string += '\tmode  = %s\n' % self.mode
        string += '\tfold = %s\n' % self.fold
        string += '\tcsv   = %s\n' % str(self.csv)
        string += '\t\tlen  = %5d\n' % length

        return string

    def __len__(self):
        return len(self.df)

    def fold_name(self):
        return self.fold[0].split('_')[1]

    def num_classes(self):
        return self.num_classes

    def preprocess(self):
        lbl_encoder = LabelEncoder()
        self.df['label'] = lbl_encoder.fit_transform(self.df['label_group'])

        self.num_classes = self.df['label'].nunique()

        # https://www.kaggle.com/moeinshariatnia/indonesian-distilbert-finetuning-with-arcmargin
        # title_lengths = self.df['title'].apply(lambda x: len(x.split(" "))).to_numpy()
        # print(f"MIN words: {title_lengths.min()}, MAX words: {title_lengths.max()}")
        # self.max_length = title_lengths.max()


class ProductTextTestDataset(data.Dataset):
    def __init__(self, data_dir, csv, tokenizer):
        self.data_dir = data_dir
        self.csv = csv
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH
        self.num_classes = NUM_CLASS

        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in self.csv])
        texts = list(self.df['title'].apply(lambda o: str(o)).values)
        self.encodings = tokenizer(texts,
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length)

    def __getitem__(self, index):
        # putting each tensor in front of the corresponding key from the tokenizer
        # HuggingFace tokenizers give you whatever you need to feed to the corresponding model
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.df)


class WordVectorDataset(data.Dataset):
    def __init__(self, data_dir, csv, file, model, mode):
        self.data_dir = data_dir
        self.csv = csv
        self.file = file
        self.model = model
        self.mode = mode

        df = pd.read_csv(os.path.join(data_dir, csv))
        self.posting_id = df['posting_id'].values.tolist()
        self.labels = df['label'].values.tolist()

        self.features = self.sentence_vector()

    def sentence_vector(self):
        features = []

        f = open(self.file, 'r')
        lines = f.readlines()
        for line in lines:
            sent_vect = []
            for sent in line.strip().split(" "):
                try:
                    vect = self.model.wv[sent]
                except:
                    continue
                sent_vect.append(vect)
            features.append(np.mean(np.stack(sent_vect), axis=0))

        f.close()

        return features

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.posting_id[index]

    def __len__(self):
        return len(self.features)

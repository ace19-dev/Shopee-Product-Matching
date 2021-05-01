import re
import string
import codecs
import emoji
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
added_stopwords = ['grosir', 'cod', 'diskon', 'starseller']

NUM_CLASS = 11014
MAX_LENGTH = 24

# factory = StemmerFactory()

# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/execution?select=embeddings.zip
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

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
            # try:
            #     float(word)
            # except:
            #     clean_words.append(word)

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


# TODO: title 에 맞게 수정 필요.
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


# TODO:
# https://stackoverflow.com/questions/51217909/removing-all-emojis-from-text
# def clean_emojjs(text):
# text2 = repr(text)
# text3 = codecs.decode(text2, 'unicode_escape')
# # TODO: how to use below re pattern
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


# def keep_only_alphanumeric_space(text):
#     return re.sub(r'[^A-Za-z0-9]+', '', text)


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

        self.df['lowered_title'] = self.df['title'].apply(lambda x: x.lower())
        self.df['clean_special_chars_title'] = self.df['lowered_title'].apply(
            lambda x: clean_special_chars(x))
        self.df['clean_emojjs_title'] = self.df['clean_special_chars_title'].apply(lambda x: clean_emojjs(x))
        self.df['title_clean'] = self.df['clean_emojjs_title'].apply(lambda x: clean_text(x))
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
        self.max_length = 48
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
